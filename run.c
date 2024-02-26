/* Copyright (c) 2024 NinjaLABO https://ninjalabo.ai */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <dirent.h>
#include <sys/stat.h>

#define IMAGE_SZ (28*28)

typedef struct {
	int nclasses;		// the number of classes
	int nconv;		// the number of convolutional layers
	int nlinear;		// the number of linear layers
} ModelConfig;

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
} LinearConfig;

typedef struct {
	int ksize;		// kernel size
	int stride;
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
} ConvConfig;

typedef struct {
	float *x;		// buffer to store the output of a layer (9*28*28,)
	float *x2;		// buffer to store the output of a layer (4*28*28,)
} Runstate;

typedef struct {
	ModelConfig model_config;
	LinearConfig *linear_config;	// linear layers' config
	ConvConfig *conv_config;	// convolutional layers' config
	float *parameters;	// array of all weigths and biases
	Runstate state;		// the current state in the forward pass
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

static void malloc_run_state(Runstate * s)
{
	s->x = calloc(9 * 28 * 28, sizeof(float));
	s->x2 = calloc(4 * 28 * 28, sizeof(float));
}

static void free_run_state(Runstate * s)
{
	free(s->x);
	free(s->x2);
}

static void read_checkpoint(char *path, ModelConfig * config, ConvConfig ** cl,
		     LinearConfig ** ll, float **parameters, int *fd,
		     float **data, size_t *file_size)
{
	// The data inside the file should follow the order: ModelConfig -> ConvConfig -> LinearConfig -> parameters (first CNN parameters then FC parameters)
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "Couldn't open file %s\n", path);
		exit(EXIT_FAILURE);
	}
	// read model config
	if (fread(config, sizeof(ModelConfig), 1, file) != 1) {
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
	*cl = (ConvConfig *) (*data + sizeof(ModelConfig) / sizeof(float));
	*ll =
	    (LinearConfig *) (*data +
			      (config->nconv * sizeof(ConvConfig) +
			       sizeof(ModelConfig)) / sizeof(float));
	// memory map weights and biases
	int header_size =
	    sizeof(ModelConfig) + config->nconv * sizeof(ConvConfig) +
	    config->nlinear * sizeof(LinearConfig);
	*parameters = *data + header_size / sizeof(float);	// position the parameters pointer to the start of the parameter data
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

static void linear(float *xout, float *x, float *p, int in, int out)
{
	// linear layer: w(out,in) @ x (in,) + b(out,) -> xout (out,)
	// by far the most amount of time is spent inside this little function
	int i;
	float *w = p;
	float *b = p + in * out;
	//#pragma omp parallel for private(i)
	for (i = 0; i < out; i++) {
		float val = 0.0f;
		for (int j = 0; j < in; j++) {
			val += w[i * in + j] * x[j];
		}
		xout[i] = val + b[i];
	}
}

static void linear_with_relu(float *xout, float *x, float *p, int in, int out)
{
	// linear layer with ReLU activation: w(out,in) @ x (in,) + b(out,) -> xout (out,)
	int i;
	float *w = p;
	float *b = p + in * out;
	//#pragma omp parallel for private(i)
	for (i = 0; i < out; i++) {
		float val = 0.0f;
		for (int j = 0; j < in; j++) {
			val += w[i * in + j] * x[j];
		}
		xout[i] = fmax(val + b[i], 0.0f);
	}
}

static void matmul_conv_with_relu(float *xout, float *x, float *p, int nchannels,
			   int in, int out)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(chan,) -> xout (nchannels,out)
	int c;
	float *w = p;
	float *b = p + nchannels * in;
	//#pragma omp parallel for private(c)
	for (c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val = 0.0f;
			for (int j = 0; j < in; j++) {
				val += w[c * in + j] * x[j * out + i];
			}
			xout[c * out + i] = fmax(val + b[c], 0.0f);
		}
	}
}

static float im2col_get_pixel(float *im, int height, int width, int row, int col,
		       int channel, int pad)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[col + width * (row + height * channel)];
}

static void im2col_cpu(float *col, float *im, int nchannels, int height, int width,
		int ksize, int stride, int pad)
{
	// im (nchannels, height, width) -> col (col_size, out_height * out_width)
	int c, h, w;
	int out_height = (height + 2 * pad - ksize) / stride + 1;
	int out_width = (width + 2 * pad - ksize) / stride + 1;

	int col_size = nchannels * ksize * ksize;
	for (c = 0; c < col_size; c++) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int channel = c / ksize / ksize;
		for (h = 0; h < out_height; h++) {
			for (w = 0; w < out_width; w++) {
				int input_row = h_offset + h * stride;
				int input_col = w_offset + w * stride;
				int col_index =
				    (c * out_height + h) * out_width + w;
				col[col_index] =
				    im2col_get_pixel(im, height, width,
						     input_row, input_col,
						     channel, pad);
			}
		}
	}
}

static void maxpool(float *x, int height, int width, int nchannels, int ksize)
{
	int out_height = height / ksize;
	int out_width = width / ksize;
	for (int c = 0; c < nchannels; c++) {
		int xout_idx = c * out_height * out_width;	// start index for x
		for (int i = 0; i < out_height; i++) {
			for (int j = 0; j < out_width; j++) {
				float cmax = 0;
				int x_idx = c * height * width + 2 * (i * width + j);	// start index for x
				for (int ki = 0; ki < ksize; ki++) {
					for (int kj = 0; kj < ksize; kj++) {
						cmax =
						    fmax(cmax,
							 x[x_idx + ki * width +
							   kj]);
					}
				}
				x[xout_idx + i * out_width + j] = cmax;
			}
		}
	}
}

static void normalize(float *xout, uint8_t * image)
{
	// normalize values [0, 255] -> [-1, 1]
	for (int i = 0; i < IMAGE_SZ; i++) {
		xout[i] = ((float)image[i] / 255 - 0.5) / 0.5;
	}
}

static void softmax(float *x, int size)
{
	// find max value (for numerical stability)
	float max_val = x[0];
	for (int i = 1; i < size; i++) {
		if (x[i] > max_val) {
			max_val = x[i];
		}
	}
	// exp and sum
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		x[i] = expf(x[i] - max_val);
		sum += x[i];
	}
	// normalize
	for (int i = 0; i < size; i++) {
		x[i] /= sum;
	}
}

static void read_mnist_image(char *path, uint8_t * image)
{
	size_t bytes;
	FILE *file = fopen(path, "rb");
	if (!file) {
		perror("Error opening file");
		exit(EXIT_FAILURE);
	}
	bytes = fread(image, sizeof(uint8_t), IMAGE_SZ, file);
	if (!bytes) {
		perror("Error reading file");
		exit(EXIT_FAILURE);
	}
	fclose(file);
}

#ifdef DEBUG
// sanity check function for writing tensors, e.g., it can be used to evaluate values after a specific layer.
static void write_tensor(float *x, int size)
{
	FILE *f = fopen("tensor.txt", "w");
	for (int i = 0; i < size; i++)
		fprintf(f, "%f\n", x[i]);
	fclose(f);
}
#endif

static void forward(Model * m, uint8_t * image)
{
	ConvConfig *cl = m->conv_config;
	LinearConfig *ll = m->linear_config;
	float *p = m->parameters;
	Runstate *s = &m->state;
	float *x = s->x;
	float *x2 = s->x2;

	normalize(x2, image);
	im2col_cpu(x, x2, cl[0].ic, 28, 28, cl[0].ksize, cl[0].stride,
		   cl[0].pad);
	matmul_conv_with_relu(x2, x, p + cl[0].offset, cl[0].oc,
			      cl[0].ic * cl[0].ksize * cl[0].ksize, 28 * 28);
	maxpool(x2, 28, 28, cl[0].oc, 2);
	im2col_cpu(x, x2, cl[1].ic, 14, 14, cl[1].ksize, cl[1].stride,
		   cl[1].pad);
	matmul_conv_with_relu(x2, x, p + cl[1].offset, cl[1].oc,
			      cl[1].ic * cl[1].ksize * cl[1].ksize, 14 * 14);
	maxpool(x2, 14, 14, cl[1].oc, 2);
	linear_with_relu(x, x2, p + ll[0].offset, ll[0].in, ll[0].out);
	linear(x2, x, p + ll[1].offset, ll[1].in, ll[1].out);
	softmax(x2, ll[1].out);
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
	uint8_t *image = malloc(IMAGE_SZ);
	for (int i = 2; i < argc; i++) {
		image_path = argv[i];
		read_mnist_image(image_path, image);
		forward(&model, image);	// output (nclasses,) is stored in model.state.x2
		for (int j = 0; j < model.model_config.nclasses; j++) {
			printf("%f\t", model.state.x2[j]);
		}
		printf("\n");
	}

	free(image);
	free_model(&model);
	return 0;
}
