/* Copyright (c) 2024 NinjaLABO https://ninjalabo.ai */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <string.h>

#include "func.h"

#define IMAGE_SZ (3 * 224 * 224)	// model input image size (all images are resized to this)
#define IM2COL_MAX_SZ (3 * 49 * 112 * 112)	// maximum array size after im2col
#define IM2COL_SECOND_MAX_SZ (64 * 9 * 56 * 56)	// second maximum array size after im2col
#define OUTPUT_MAX_SZ (64 * 112 * 112)	// maximum output size of layers during forward pass

typedef struct {
	float *x;		// buffer to store the input (28*28,)
	float *x2;		// buffer to store the output of a layer (25*28*28,)
	float *x3;		// buffer to store the output of a layer (9*28*28,)
} Runstate;

typedef struct {
	ModelConfig model_config;
	ConvConfig *conv_config;	// convolutional layers' config
	LinearConfig *linear_config;	// linear layers' config
	BnConfig *bn_config;		// batchnorm layers' config
	float *parameters;	// array of all weigths and biases
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
}

static void free_run_state(Runstate * s)
{
	free(s->x);
	free(s->x2);
	free(s->x3);
}

static void read_checkpoint(char *path, ModelConfig * mc, ConvConfig ** cc,
			    LinearConfig ** lc, BnConfig ** bc,
			    float **parameters, int *fd, float **data,
			    size_t *file_size)
{
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "Couldn't open file %s\n", path);
		exit(EXIT_FAILURE);
	}
	// read model config
	if (fread(mc, sizeof(ModelConfig), 1, file) != 1) {
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
	*cc = (ConvConfig *) (*data + sizeof(ModelConfig) / sizeof(float));
	*lc = (LinearConfig *) (*cc + mc->nconv);
	*bc = (BnConfig *) (*lc + mc->nlinear);
	// memory map weights and biases
	*parameters = (float *) (*bc + mc->nbn);
}

static void build_model(Model * m, char *checkpoint_path)
{
	// read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, &m->model_config, &m->conv_config,
			&m->linear_config, &m->bn_config, &m->parameters,
			&m->fd, &m->data, &m->file_size);
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
// sanity check function for writing tensors, e.g., it can be used to evaluate values after a specific layer.
static void write_tensor(float *x, int size)
{
	FILE *f = fopen("test1.txt", "w");
	for (int i = 0; i < size; i++)
		fprintf(f, "%f\n", x[i]);
	fclose(f);
}
#endif

static void basic_block(float *x, float *x2, float *x3, float *p,
			ConvConfig *cc, int *h, int *w, int *i,
			bool downsample)
{
	int h_prev = *h;
	int w_prev = *w;

	im2col(x3, x, cc[*i], h, w);
	conv(x, x3, p, cc[*i], *h, *w);
	relu(x, cc[*i].oc * (*h) * (*w));
	im2col(x3, x, cc[*i + 1], h, w);
	conv(x, x3, p, cc[*i + 1], *h, *w);
	// perform downsampling for skip connection
	if (downsample) {
		im2col(x3, x2, cc[*i + 2], &h_prev, &w_prev);
		conv(x2, x3, p, cc[*i + 2], *h, *w);
	}

	int oc = downsample ? cc[*i + 2].oc : cc[*i + 1].oc;
	int size = oc * (*h) * (*w) * sizeof(float);
	*i += downsample ? 3 : 2;
	matadd(x, x2, oc * (*h) * (*w));
	relu(x, oc * (*h) * (*w));
	memcpy(x2, x, size);
}

 static void bottleneck(float *x, float *x2, float *x3, float *p,
			ConvConfig *cc, int *h, int *w, int *i,
			bool downsample)
{
	int h_prev = *h;
	int w_prev = *w;

	im2col(x3, x, cc[*i], h, w);
	conv(x, x3, p, cc[*i], *h, *w);
	relu(x, cc[*i].oc * (*h) * (*w));
	im2col(x3, x, cc[*i + 1], h, w);
	conv(x, x3, p, cc[*i + 1], *h, *w);
	relu(x, cc[*i + 1].oc * (*h) * (*w));
	im2col(x3, x, cc[*i + 2], h, w);
	conv(x, x3, p, cc[*i + 2], *h, *w);
	// perform downsampling for skip connection
	if (downsample) {
		im2col(x3, x2, cc[*i + 3], &h_prev, &w_prev);
		conv(x2, x3, p, cc[*i + 3], *h, *w);
	}

	int oc = downsample ? cc[*i + 3].oc : cc[*i + 2].oc;
	int size = oc * (*h) * (*w) * sizeof(float);
	*i += downsample ? 4 : 3;
	matadd(x, x2, oc * (*h) * (*w));
	relu(x, oc * (*h) * (*w));
	memcpy(x2, x, size);
}

 static void sequential_block(float *x, float *x2, float *x3, float *p,
			      ConvConfig *cc, int *h, int *w, int *i, int n,
			      bool downsample)
{
	if (downsample) {
		basic_block(x, x2, x3, p, cc, h, w, i, true);
		for (int j = 0; j < n - 1; j++) {
			basic_block(x, x2, x3, p, cc, h, w, i, false);
		}
	} else {
		for (int j = 0; j < n; j++) {
			basic_block(x, x2, x3, p, cc, h, w, i, false);
		}
	}
}

 static void sequential_bottleneck(float *x, float *x2, float *x3, float *p,
				   ConvConfig *cc, int *h, int *w, int *i,
				   int n, bool downsample)
{
	if (downsample) {
		bottleneck(x, x2, x3, p, cc, h, w, i, true);
		for (int j = 0; j < n - 1; j++) {
			bottleneck(x, x2, x3, p, cc, h, w, i, false);
		}
	} else {
		for (int j = 0; j < n; j++) {
			bottleneck(x, x2, x3, p, cc, h, w, i, false);
		}
	}
}

 static void head(float *x, float *x2, float *image, float *p, ConvConfig *cc,
		  int *h, int *w)
{
	im2col(x, image, cc[0], h, w);
	conv(x2, x, p, cc[0], *h, *w);
	relu(x2, cc[0].oc * (*h) * (*w));
	maxpool(x, x2, h, w, cc[0].oc, 3, 2, 1);
	memcpy(x2, x, cc[0].oc * (*h) * (*w) * sizeof(float));
}

 static void tail(float *x, float *x2, float *x3, float *image, float *p,
		  ConvConfig *cc, LinearConfig *lc, BnConfig *bc, int h, int w,
		  int i)
{
	int h_prev = h;
	int w_prev = w;
	maxpool(x2, x, &h, &w, cc[i - 1].oc, h, 1, 0); // by expecting h = w
	avgpool(x2 + cc[i - 1].oc, x, &h_prev, &w_prev, cc[i - 1].oc, h_prev,
		1, 0);
	batchnorm(x, x2, p, bc[0], 1);
	linear(x2, x, p, lc[0]);
	relu(x2, lc[0].out);
	batchnorm(x, x2, p, bc[1], 1);
	linear(x2, x, p, lc[1]);
	softmax(x2, lc[1].out);
	memcpy(x, x2, lc[1].out * sizeof(float));
}

 static void resnet18(float *x, float *x2, float *x3, float *image, float *p,
		      ConvConfig *cc, LinearConfig *lc, BnConfig *bc, int h,
		      int w)
{
	head(x, x2, image, p, cc, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	// 2 BasicBlocks, no downsampling
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 2, false);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 2, true);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 2, true);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 2, true);
	// fine tuned layers
	tail(x, x2, x3, image, p, cc, lc, bc, h, w, i);
}

 static void resnet34(float *x, float *x2, float *x3, float *image, float *p,
		      ConvConfig *cc, LinearConfig *lc, BnConfig *bc, int h,
		      int w)
{
	head(x, x2, image, p, cc, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	// 3 BasicBlocks, no downsampling
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 3, false);
	// 4 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 4, true);
	// 6 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 6, true);
	// 3 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 3, true);
	// fine tuned layers
	tail(x, x2, x3, image, p, cc, lc, bc, h, w, i);
}

 static void resnet50(float *x, float *x2, float *x3, float *image, float *p,
		      ConvConfig *cc, LinearConfig *lc, BnConfig *bc, int h,
		      int w)
{
	head(x, x2, image, p, cc, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	// 3 Bottlenecks, with one downsampling block
	sequential_bottleneck(x, x2, x3, p, cc, &h, &w, &i, 3, true);
	// 4 Bottlenecks, with one downsampling block
	sequential_bottleneck(x, x2, x3, p, cc, &h, &w, &i, 4, true);
	// 6 Bottlenecks, with one downsampling block
	sequential_bottleneck(x, x2, x3, p, cc, &h, &w, &i, 6, true);
	// 3 Bottlenecks, with one downsampling block
	sequential_bottleneck(x, x2, x3, p, cc, &h, &w, &i, 3, true);
	// fine tuned layers
	tail(x, x2, x3, image, p, cc, lc, bc, h, w, i);
}

static void forward(Model *m, float *image, int model_size)
{
	ConvConfig *cc = m->conv_config;
	LinearConfig *lc = m->linear_config;
	BnConfig *bc = m->bn_config;
	float *p = m->parameters;
	Runstate *s = &m->state;
	float *x = s->x;
	float *x2 = s->x2;
	float *x3 = s->x3;

	int h = 224; // height
	int w = 224; // width

	if (model_size == 18)
		resnet18(x, x2, x3, image, p, cc, lc, bc, h, w);
	else if (model_size == 34)
		resnet34(x, x2, x3, image, p, cc, lc, bc, h, w);
	else if (model_size == 50)
		resnet50(x, x2, x3, image, p, cc, lc, bc, h, w);
	else {
		fprintf(stderr, "Invalid structure size\n");
		exit(EXIT_FAILURE);
	}
}

static void error_usage()
{
	fprintf(stderr, "Usage:   run <model size> <model> <image>\n");
	fprintf(stderr, "Example: run model.bin image1 image2 ... imageN\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	char *model_path = NULL;
	char *image_path = NULL;

	// read images and model path, then outputs the probability distribution for the given images.
	if (argc < 4) {
		error_usage();
	}

	int model_size = atoi(argv[1]);
	model_path = argv[2];
	Model model;
	build_model(&model, model_path);
	float *image = malloc(IMAGE_SZ * sizeof(float));
	for (int i = 3; i < argc; i++) {
		image_path = argv[i];
		read_imagenette_image(image_path, &image);

		forward(&model, image, model_size);
		for (int j = 0; j < model.model_config.nclasses; j++) {
			printf("%f\t", model.state.x[j]);
		}
		printf("\n");
	}

	free(image);
	free_model(&model);
	return 0;
}
