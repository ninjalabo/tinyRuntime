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
	float *x;		// buffer to store the input
	float *x2;		// buffer to store the output of a layer
	float *x3;		// buffer to store the output of a layer
} Runstate;

typedef struct {
	ModelConfig model_config;
	ConvConfig *conv_config;	// convolutional layers' config
	LinearConfig *linear_config;	// linear layers' config
	float *parameters;	// array of all weigths and biases
	Runstate state;		// the current state in the forward pass
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

static void malloc_run_state(Runstate * s)
{
	s->x = calloc(batch_size * IM2COL_MAX_SZ, sizeof(float));
	s->x2 = calloc(batch_size * OUTPUT_MAX_SZ, sizeof(float));
	s->x3 = calloc(batch_size * IM2COL_SECOND_MAX_SZ, sizeof(float));
}

static void free_run_state(Runstate * s)
{
	free(s->x);
	free(s->x2);
	free(s->x3);
}

static void read_checkpoint(char *path, ModelConfig * mc, ConvConfig ** cc,
			    LinearConfig ** lc, float **parameters, int *fd,
			    float **data, size_t *file_size)
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
	// memory map weights and biases
	*parameters = (float *)(*lc + mc->nlinear);	// position the pointer to the start of the parameter data
}

static void build_model(Model * m, char *checkpoint_path)
{
	// read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, &m->model_config, &m->conv_config,
			&m->linear_config, &m->parameters, &m->fd, &m->data,
			&m->file_size);
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

static void forward(Model * m, float *images)
{
	ConvConfig *cc = m->conv_config;
	LinearConfig *lc = m->linear_config;
	float *p = m->parameters;
	Runstate *s = &m->state;
	float *x = s->x;
	float *x2 = s->x2;
	float *x3 = s->x3;

	int h = 224;		// height
	int w = 224;		// width
	int h_prev;		// buffer to store previous height for skip connection
	int w_prev;		// buffer to store previous width for skip connection

	im2col(x, images, cc[0], &h, &w);
	conv(x2, x, p, cc[0], h, w);
	relu(x, cc[0].oc * h * w);
	maxpool(x, x2, &h, &w, cc[0].oc, 3, 2, 1);
	memcpy(x2, x, batch_size * cc[0].oc * h * w * sizeof(float));

	// block 1.1 and 1.2
	for (int i = 1; i < 4; i += 2) {
		im2col(x3, x, cc[i], &h, &w);
		conv(x, x3, p, cc[i], h, w);
		relu(x, cc[i].oc * h * w);
		im2col(x3, x, cc[i + 1], &h, &w);
		conv(x, x3, p, cc[i + 1], h, w);
		// skip connection, no change
		matadd(x, x2, cc[i + 1].oc * h * w);
		relu(x, cc[i + 1].oc * h * w);
		memcpy(x2, x,
		       batch_size * cc[i + 1].oc * h * w * sizeof(float));
	}

	// block 2-4
	for (int i = 5; i < m->model_config.nconv - 4; i += 5) {
		// block i.1
		h_prev = h;
		w_prev = w;

		im2col(x3, x, cc[i], &h, &w);
		conv(x, x3, p, cc[i], h, w);
		relu(x, cc[i].oc * h * w);
		im2col(x3, x, cc[i + 1], &h, &w);
		conv(x, x3, p, cc[i + 1], h, w);
		// skip connection, change in stride
		im2col(x3, x2, cc[i + 2], &h_prev, &w_prev);
		conv(x2, x3, p, cc[i + 2], h, w);
		matadd(x, x2, cc[i + 2].oc * h * w);
		relu(x, cc[i + 2].oc * h * w);
		memcpy(x2, x,
		       batch_size * cc[i + 2].oc * h * w * sizeof(float));

		// block i.2
		im2col(x3, x, cc[i + 3], &h, &w);
		conv(x, x3, p, cc[i + 3], h, w);
		relu(x, cc[i + 3].oc * h * w);
		im2col(x3, x, cc[i + 4], &h, &w);
		conv(x, x3, p, cc[i + 4], h, w);
		// skip connection, no change
		matadd(x, x2, cc[i + 4].oc * h * w);
		relu(x, cc[i + 4].oc * h * w);
		// the final block output doesn't need to be copied
		if (i < 11) {
			memcpy(x2, x,
			       batch_size * cc[i + 4].oc * h * w *
			       sizeof(float));
		}
	}

	avgpool(x2, x, &h, &w, cc[m->model_config.nconv - 1].oc, h, 1, 0);
	linear(x, x2, p, lc[0]);
	softmax(x, lc[0].out);
}

static void error_usage()
{
	fprintf(stderr, "Usage:   run <model> <image>\n");
	fprintf(stderr, "Example: run model.bin image1 image2 ... imageN\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	char *model_path = NULL;
	char **image_paths = NULL;

	// read images and model path, then outputs the probability distribution for the given images.
	if (argc < 3) {
		error_usage();
	}
	// set global variable batch size
	char* bs_env = getenv("BS");
	int bs = (bs_env != NULL) ? atoi(bs_env) : 1;
	if (bs <= 0) {
		printf("Invalid batch size\n");
		exit(EXIT_FAILURE);
	}
	batch_size = bs;

	model_path = argv[1];
	Model model;
	build_model(&model, model_path);
	float *images = malloc(batch_size * IMAGE_SZ * sizeof(float));
	image_paths = &argv[2];
	int niter = (argc - 2) / batch_size;
	int nclasses = model.model_config.nclasses;
	for (int i = 0; i < niter; i++) {
		read_imagenette_image(&image_paths[i * batch_size], images);
		forward(&model, images);	// output (nclass,) is stored in model.state.x
		for (int bs = 0; bs < batch_size; bs++) {
			for (int j = 0; j < nclasses; j++) {
				printf("%f\t",
				       model.state.x[bs * nclasses + j]);
			}
			printf("\n");
		}
	}
	// process the remaining images
	batch_size = (argc - 2) % batch_size;
	if (batch_size != 0) {
		read_imagenette_image(&image_paths[niter * batch_size], images);
		forward(&model, images);
		for (int bs = 0; bs < batch_size; bs++) {
			for (int j = 0; j < model.model_config.nclasses; j++) {
				printf("%f\t", model.state.x[j]);
			}
			printf("\n");
		}
	}

	free(images);
	free_model(&model);
	return 0;
}
