/* Copyright (c) 2024 NinjaLABO https://ninjalabo.ai */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

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

static void forward(Model * m, float *image)
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

	im2col(x, image, &h, &w, cc[0]);
	conv(x2, x, p, cc[0], h * w, true);
	maxpool(x, x2, &h, &w, cc[0].oc, 3, 2, 1);
	memcpy(x2, x, cc[0].oc * h * w * sizeof(float));

	// block 1.1 and 1.2
	for (int i = 1; i < 4; i += 2) {
		im2col(x3, x, &h, &w, cc[i]);
		conv(x, x3, p, cc[i], h * w, true);
		im2col(x3, x, &h, &w, cc[i + 1]);
		conv(x, x3, p, cc[i + 1], h * w, false);
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

		im2col(x3, x, &h, &w, cc[i]);
		conv(x, x3, p, cc[i], h * w, true);
		im2col(x3, x, &h, &w, cc[i + 1]);
		conv(x, x3, p, cc[i + 1], h * w, false);
		// skip connection, change in stride
		im2col(x3, x2, &h_prev, &w_prev, cc[i + 2]);
		conv(x2, x3, p, cc[i + 2], h * w, false);
		matadd(x, x2, cc[i + 2].oc * h * w);
		relu(x, cc[i + 2].oc * h * w);
		memcpy(x2, x, cc[i + 2].oc * h * w * sizeof(float));

		// block i.2
		im2col(x3, x, &h, &w, cc[i + 3]);
		conv(x, x3, p, cc[i + 3], h * w, true);
		im2col(x3, x, &h, &w, cc[i + 4]);
		conv(x, x3, p, cc[i + 4], h * w, false);
		// skip connection, no change
		matadd(x, x2, cc[i + 4].oc * h * w);
		relu(x, cc[i + 4].oc * h * w);
		// the final block output doesn't need to be copied
		if (i < 11) {
			memcpy(x2, x, cc[i + 4].oc * h * w * sizeof(float));
		}
	}

	avgpool(x2, x, &h, &w, cc[m->model_config.nconv - 1].oc, h, 1, 0);
	linear(x, x2, p, lc[0], false);
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
