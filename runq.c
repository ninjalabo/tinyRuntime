/* Copyright (c) 2024 NinjaLABO https://ninjalabo.ai */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <string.h>

#include "func.h"
#include "func_q.h"

#define IMAGE_SZ (3 * 224 * 224)	// model input image size (all images are resized to this)
#define IM2COL_MAX_SZ (3 * 49 * 112 * 112)	// maximum array size after im2col
#define IM2COL_SECOND_MAX_SZ (64 * 9 * 56 * 56)	// second maximum array size after im2col
#define OUTPUT_MAX_SZ (64 * 112 * 112)	// maximum output size of layers during forward pass

typedef struct {
	float *x;		// buffer to store the output of a layer (IMAGE_SZ,)
	float *x2;
	float *x3;
	QuantizedTensor xq;	// buffer for quantized arrays
} Runstate;

typedef struct {
	ModelConfigQ model_config;
	ConvConfigQ *conv_config;	// convolutional layers' config
	LinearConfigQ *linear_config;	// linear layers' config
	BnConfig *bn_config;	// batchnorm layers' config
	int8_t *parameters;	// array of all weigths and biases
	float *scaling_factors;	// array of all scaling factors
	Runstate state;		// the current state in the forward pass
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

static void malloc_run_state(Runstate * s)
{
	s->x = calloc(IM2COL_MAX_SZ, sizeof(float));
	s->x2 = calloc(OUTPUT_MAX_SZ, sizeof(float));
	s->x3 = calloc(IM2COL_SECOND_MAX_SZ, sizeof(float));
	// FIX: too much memory allocated for `xq.s`. Should be divided by group size
	s->xq = (QuantizedTensor) {
	.q = calloc(IM2COL_MAX_SZ, sizeof(int8_t)),.s =
		    calloc(IM2COL_MAX_SZ, sizeof(float))};
}

static void free_run_state(Runstate * s)
{
	free(s->x);
	free(s->x2);
	free(s->x3);
	free(s->xq.q);
	free(s->xq.s);
}

static void read_checkpoint(char *path, ModelConfigQ * mc, ConvConfigQ ** cc,
			    LinearConfigQ ** lc, BnConfig ** bc,
			    int8_t ** parameters, float **scaling_factors,
			    int *fd, float **data, size_t *file_size)
{
	// The data inside the file should follow the order: ModelConfigQ -> ConvConfigQ -> LinearConfigQ -> parameters (first CNN parameters then FC parameters)
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "Couldn't open file %s\n", path);
		exit(EXIT_FAILURE);
	}
	// read model config
	if (fread(mc, sizeof(ModelConfigQ), 1, file) != 1) {
		exit(EXIT_FAILURE);
	}
	// figure out the file size
	fseek(file, 0, SEEK_END);	// move file pointer to end of file
	*file_size = ftell(file);	// get the file size, in bytes
	fclose(file);
	// memory map layers' config
	*fd = open(path, O_RDONLY);	// open in read only mode
	if (*fd == -1) {
		fprintf(stderr, "open failed!\n");
		exit(EXIT_FAILURE);
	}
	*data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
	if (*data == MAP_FAILED) {
		fprintf(stderr, "mmap failed!\n");
		exit(EXIT_FAILURE);
	}
	*cc = (ConvConfigQ *) (*data + sizeof(ModelConfigQ) / sizeof(float));
	*lc = (LinearConfigQ *) (*cc + mc->nconv);
	*bc = (BnConfig *) (*lc + mc->nlinear);
	// memory map weights, biases and scaling factors
	*parameters = (int8_t *) (*bc + mc->nbn);
	*scaling_factors = (float *)(*parameters + mc->nparameters);
}

static void build_model(Model * m, char *checkpoint_path)
{
	// read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, &m->model_config, &m->conv_config,
			&m->linear_config, &m->bn_config, &m->parameters,
			&m->scaling_factors, &m->fd, &m->data, &m->file_size);
	// allocate the RunState buffers
	malloc_run_state(&m->state);
}

static void free_model(Model * m)
{
	// close the memory mapping
	if (m->data != MAP_FAILED) {
		munmap(m->data, m->file_size);
	}
	if (m->fd != -1) {
		close(m->fd);
	}
	// free the RunState buffers
	free_run_state(&m->state);
}

static void read_imagenette_image(char *path, float **image)
{
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "Couldn't open file %s\n", path);
		exit(EXIT_FAILURE);
	}
	if (fread(*image, sizeof(float), IMAGE_SZ, file) != IMAGE_SZ) {
		printf("Image read failed");
		exit(EXIT_FAILURE);
	}
	fclose(file);
}

#ifdef DEBUG
// sanity check function for writing tensors,
// e.g., it can be used to evaluate values after a specific layer.
static void write_tensor(float *x, int size)
{
	FILE *f = fopen("test2.txt", "w");
	for (int i = 0; i < size; i++)
		fprintf(f, "%f\n", x[i]);
	fclose(f);
}

static void write_qtensor(int8_t * x, int size)
{
	FILE *f = fopen("testq.txt", "w");
	for (int i = 0; i < size; i++)
		fprintf(f, "%d\n", x[i]);
	fclose(f);
}

void dequantize(QuantizedTensor * qx, float *x, int size, int gs)
{
	for (int i = 0; i < size; i++) {
		x[i] = qx->q[i] * qx->s[i / gs];
	}
}

void dequantize2d(QuantizedTensor * qx, float *x, int nrows, int ncols, int gs)
{
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			x[i * ncols + j] =
			    qx->q[i * ncols + j] * qx->s[(j * nrows + i) / gs];
		}
	}
}

#endif

static void forward(Model * m, float *image)
{
	ConvConfigQ *cc = m->conv_config;
	LinearConfigQ *lc = m->linear_config;
	BnConfig *bc = m->bn_config;
	int8_t *p = m->parameters;
	float *sf = m->scaling_factors;
	Runstate *s = &m->state;
	float *x = s->x;
	float *x2 = s->x2;
	float *x3 = s->x3;
	QuantizedTensor xq = s->xq;

	int h = 224;		// height
	int w = 224;		// width
	int h_prev;		// buffer to store previous height for skip connection
	int w_prev;		// buffer to store previous width for skip connection

	im2col_q(x, image, cc[0], &h, &w);
	quantize2d(&xq, x, cc[0], h * w);
	conv_q(x2, &xq, p, sf, cc[0], h, w);
	batchnorm(x, x2, sf, bc[0], h, w);
	relu(x, bc[0].ic * h * w);
	maxpool(x2, x, &h, &w, cc[0].oc, 3, 2, 1);
	memcpy(x, x2, cc[0].oc * h * w * sizeof(float));

	// block 1.1 and 1.2
	for (int i = 1; i < 4; i += 2) {
		im2col_q(x3, x, cc[i], &h, &w);
		quantize2d(&xq, x3, cc[i], h * w);
		conv_q(x, &xq, p, sf, cc[i], h, w);
		batchnorm(x3, x, sf, bc[i], h, w);
		relu(x3, bc[i].ic * h * w);
		im2col_q(x, x3, cc[i + 1], &h, &w);
		quantize2d(&xq, x, cc[i + 1], h * w);
		conv_q(x3, &xq, p, sf, cc[i + 1], h, w);
		batchnorm(x, x3, sf, bc[i + 1], h, w);
		// skip connection, no change
		matadd(x, x2, cc[i + 1].oc * h * w);
		relu(x, cc[i + 1].oc * h * w);
		memcpy(x2, x, cc[i + 1].oc * h * w * sizeof(float));
	}

	// block 2-4
	for (int i = 5; i < m->model_config.nconv - 4; i += 5) {
		// block i.1
		h_prev = h;
		w_prev = w;
		im2col_q(x3, x, cc[i], &h, &w);
		quantize2d(&xq, x3, cc[i], h * w);
		conv_q(x, &xq, p, sf, cc[i], h, w);
		batchnorm(x3, x, sf, bc[i], h, w);
		relu(x3, bc[i].ic * h * w);
		im2col_q(x, x3, cc[i + 1], &h, &w);
		quantize2d(&xq, x, cc[i + 1], h * w);
		conv_q(x3, &xq, p, sf, cc[i + 1], h, w);
		batchnorm(x, x3, sf, bc[i + 1], h, w);
		// skip connection, change in stride
		im2col_q(x3, x2, cc[i + 2], &h_prev, &w_prev);
		quantize2d(&xq, x3, cc[i + 2], h * w);
		conv_q(x2, &xq, p, sf, cc[i + 2], h, w);
		batchnorm(x3, x2, sf, bc[i + 2], h, w);
		matadd(x, x3, cc[i + 2].oc * h * w);
		relu(x, cc[i + 2].oc * h * w);
		memcpy(x2, x, cc[i + 2].oc * h * w * sizeof(float));

		// block i.2
		im2col_q(x3, x, cc[i + 3], &h, &w);
		quantize2d(&xq, x3, cc[i + 3], h * w);
		conv_q(x, &xq, p, sf, cc[i + 3], h, w);
		batchnorm(x3, x, sf, bc[i + 3], h, w);
		relu(x3, bc[i + 3].ic * h * w);
		im2col_q(x, x3, cc[i + 4], &h, &w);
		quantize2d(&xq, x, cc[i + 4], h * w);
		conv_q(x3, &xq, p, sf, cc[i + 4], h, w);
		batchnorm(x, x3, sf, bc[i + 4], h, w);
		// skip connection, no change
		matadd(x, x2, cc[i + 4].oc * h * w);
		relu(x, cc[i + 4].oc * h * w);
		// the final block output doesn't need to be copied
		if (i < 11) {
			memcpy(x2, x, cc[i + 4].oc * h * w * sizeof(float));
		}
	}

	avgpool(x2, x, &h, &w, cc[m->model_config.nconv - 1].oc, h, 1, 0);
	quantize(&xq, x2, lc[0].in, lc[0].gs_weight);
	linear_q(x, &xq, p, sf, lc[0]);
	softmax(x, lc[0].out);
}

static void error_usage()
{
	fprintf(stderr, "Usage:   runq <model> <image>\n");
	fprintf(stderr, "Example: runq modelq8.bin image1 image2 ... imageN\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	char *model_path = NULL;
	char *image_path = NULL;

	// read images and model path, then outputs the probability distribution for the given images.
	if (argc < 3) {
		error_usage();
	}

	model_path = argv[1];
	Model model;
	build_model(&model, model_path);
	float *image = malloc(IMAGE_SZ * sizeof(float));
	for (int i = 2; i < argc; i++) {
		image_path = argv[i];
		read_imagenette_image(image_path, &image);

		forward(&model, image);	// output (nclass,) is stored in model.state.x
		for (int j = 0; j < model.model_config.nclasses; j++) {
			printf("%f\t", model.state.x[j]);
		}
		printf("\n");
	}

	free(image);
	free_model(&model);
	return 0;
}
