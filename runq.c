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
	int8_t *q;		// quantized values
	float *s;		// scaling factors
} QuantizedTensor;

typedef struct {
	int nclasses;		// the number of classes
	int nconv;		// the number of convolutional layers
	int nlinear;		// the number of linear layers
	int nparameters;		// the number of parameters
} ModelConfig;

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int qoffset;		// offset for quantized parameters
	int soffset;		// offset for scaling factors
	int gs_weight;		// group size of weights
	int gs_bias;		// group size of biases
} LinearConfig;

typedef struct {
	int ksize;		// kernel size
	int stride;
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int qoffset;		// offset for quantized parameters
	int soffset;		// offset for scaling factors
	int gs_weight;		// group size of weights
	int gs_bias;		// group size of biases
} ConvConfig;

typedef struct {
	float *x;		// buffer to store the output of a layer (IMAGE_SZ,)
	float *x2;
	QuantizedTensor xq;	// buffer for quantized arrays
} Runstate;

typedef struct {
	ModelConfig model_config;
	LinearConfig *linear_config;	// linear layers' config
	ConvConfig *conv_config;	// convolutional layers' config
	int8_t *parameters;	// array of all weigths and biases
	float *scaling_factors; // array of all scaling factors
	Runstate state;		// the current state in the forward pass
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

static void malloc_run_state(Runstate *s)
{
	s->x = calloc(9 * 28 * 28, sizeof(float));
	s->x2 = calloc(9 * 28 * 28, sizeof(float));
	s->xq = (QuantizedTensor) {
	.q = calloc(9 * 28 * 28, sizeof(int8_t)),.s =
		    calloc(9 * 28 * 28 / 9, sizeof(float))};
}

static void free_run_state(Runstate *s)
{
	free(s->x);
	free(s->x2);
	free(s->xq.q);
	free(s->xq.s);
}

static void read_checkpoint(char *path, ModelConfig * config, ConvConfig ** cl,
		     LinearConfig ** ll, int8_t **parameters, float **scaling_factors,
			 int *fd, float **data, size_t *file_size)
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
	*parameters = (int8_t *) *data + header_size;	// position the parameters pointer to the start of the parameter data
	*scaling_factors = (float*) (*parameters + config->nparameters);
}

static void build_model(Model * m, char *checkpoint_path)
{
	// read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, &m->model_config, &m->conv_config,
			&m->linear_config, &m->parameters, &m->scaling_factors, &m->fd,
			&m->data, &m->file_size);
	// allocate the RunState buffers
	malloc_run_state(&m->state);
}

static void free_model(Model *m)
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

static void quantize(QuantizedTensor *qx, float *x, int n, int gs)
{
	int num_groups = n / gs;
	float Q_MAX = 127.0f;

	for (int group = 0; group < num_groups; group++) {
		// find the max absolute value in the current group
		float wmax = 0.0;
		for (int i = 0; i < gs; i++) {
			float val = fabs(x[group * gs + i]);
			if (val > wmax) {
				wmax = val;
			}
		}
		// calculate and write the scaling factor
		float scale = wmax / Q_MAX;
		qx->s[group] = scale;

		// calculate and write the quantized values
		for (int i = 0; i < gs; i++) {
			float quant_value = x[group * gs + i] / scale;	// scale
			int8_t quantized = (int8_t) round(quant_value);	// round and clamp
			qx->q[group * gs + i] = quantized;
		}
	}
}

static void quantize2d(QuantizedTensor *qx, float *x, int height, int width, int gs)
{
	// quantize each column in 2d tensor with a specified group size
	int num_groups = height / gs;
	float Q_MAX = 127.0f;

	for (int col = 0; col < width; col++) {
		for (int group = 0; group < num_groups; group++) {
			// find the max absolute value in the current group
			float wmax = 0.0;
			for (int i = 0; i < gs; i++) {
				float val = fabs(x[(group * gs + i) * width + col]);
				if (val > wmax) {
					wmax = val;
				}
			}
			// calculate and write the scaling factor
			float scale = wmax / Q_MAX;
			qx->s[col * num_groups + group] = scale;

			// calculate and write the quantized values
			for (int i = 0; i < gs; i++) {
				float quant_value = x[(group * gs + i) * width + col] / scale;	// scale
				int8_t quantized = (int8_t) round(quant_value);	// round and clamp
				qx->q[(group * gs + i) * width + col] = quantized;
			}
		}
	}
}

static void linear(float *xout, QuantizedTensor *x, int8_t *p,
			float *s, int in, int out, int gs_w, int gs_b)
{
	// linear layer: w(out,in) @ x (in,) + b(out,) -> xout (out,)
	// by far the most amount of time is spent inside this little function
	int i;
	int8_t *w = p;
	int8_t *b = p + in * out;
	float *sw = s;
	float *sb = s + in * out / gs_w;
	//#pragma omp parallel for private(i)
	for (i = 0; i < out; i++) {
		float val = 0.0f;
		int32_t ival = 0;
		// do the matmul in groups of gs_w
		for (int j = 0; j < in; j += gs_w) {
			for (int k = 0; k < gs_w; k++) {
				ival +=
				    ((int32_t) x->q[j + k]) *
				    ((int32_t) w[i * in + j + k]);
			}
			val +=
			    ((float)ival) * sw[(i * in + j) / gs_w] * x->s[j / gs_w];
			ival = 0;
		}
		xout[i] = val + ((float)b[i]) * sb[i / gs_b];
	}
}

static void linear_with_relu(float *xout, QuantizedTensor *x, int8_t *p,
				float *s, int in, int out, int gs_w, int gs_b)
{
	// linear layer with ReLU activation: w(out,in) @ x (in,) + b(out,) -> xout (out,)
	int i;
	int8_t *w = p;
	int8_t *b = p + in * out;
	float *sw = s;
	float *sb = s + in * out / gs_w;
	//#pragma omp parallel for private(i)
	for (i = 0; i < out; i++) {
		float val = 0.0f;
		int32_t ival = 0;
		// do the matmul in groups of gs_w
		for (int j = 0; j < in; j += gs_w) {
			for (int k = 0; k < gs_w; k++) {
				ival +=
				    ((int32_t) x->q[j + k]) *
				    ((int32_t) w[i * in + j + k]);
			}
			val +=
			    ((float)ival) * sw[(i * in + j) / gs_w] * x->s[j / gs_w];
			ival = 0;
		}
		xout[i] = fmax(0, val + ((float)b[i]) * sb[i / gs_b]);
	}
}

// `linear`, `linear_with_relu` and this function works only if gs <= in and in % gs = 0
static void matmul_conv_with_relu(float *xout, QuantizedTensor *x, int8_t *p,
				float *s, int nchannels, int in, int out, int gs_w, int gs_b)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(chan,) -> xout (nchannels,out)
	int c;
	int8_t *w = p;
	int8_t *b = p + nchannels * in;
	float *sw = s;
	float *sb = s + nchannels * in / gs_w;
	//#pragma omp parallel for private(c)
	for (c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val = 0.0f;
			int32_t ival = 0;
			// do the matmul in groups of gs_w
			for (int j = 0; j < in; j += gs_w) {
				for (int k = 0; k < gs_w; k++) {
					ival += ((int32_t) x->q[(j + k) * out + i]) *
					((int32_t) w[c * in + (j + k)]);
				}
				val += (float) ival * x->s[(i * in + j) / gs_w] * sw[(c * in + j) / gs_w];
			}
			xout[c * out + i] = fmax(0, val + ((float)b[c]) * sb[c / gs_b]);
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

// slightly slower than im2col + quantize2d but uses less memory as model.state.x2 is unnecessary when using this function.
// static void im2col_quantize(QuantizedTensor *col, float *im, int nchannels, int height, int width,
// 		int ksize, int stride, int pad, int gs)
// {
// 	// perform im2col operation and quantization
// 	// im (nchannels, height, width) -> col (col_size, out_height * out_width)
// 	int i, h, w;
// 	int out_height = (height + 2 * pad - ksize) / stride + 1;
// 	int out_width = (width + 2 * pad - ksize) / stride + 1;

// 	int col_size = nchannels * ksize * ksize;
// 	int num_groups = col_size / gs;
// 	float Q_MAX = 127.0f;
// 	for (h = 0; h < out_height; h++) {
// 		for (w = 0; w < out_width; w++) {
// 			for (int group = 0; group < num_groups; group++) {
// 				float wmax = 0.0;
// 				for (i = 0; i < gs; i ++) {
// 					int c = group * gs + i;
// 					int w_offset = c % ksize;
// 					int h_offset = (c / ksize) % ksize;
// 					int channel = c / ksize / ksize;
// 					int input_row = h_offset + h * stride;
// 					int input_col = w_offset + w * stride;

// 					float val = fabs(im2col_get_pixel(im, height, width,
// 							input_row, input_col, channel, pad));
// 					if (val > wmax) {
// 						wmax = val;
// 					}
// 				}
// 				// calculate and write the scaling factor
// 				float scale = wmax / Q_MAX;
// 				col->s[(h * out_width + w) * num_groups + group] = scale;

// 				// calculate and write the quantized values
// 				for (int i = 0; i < gs; i++) {
// 					int c = group * gs + i;
// 					int w_offset = c % ksize;
// 					int h_offset = (c / ksize) % ksize;
// 					int channel = c / ksize / ksize;
// 					int input_row = h_offset + h * stride;
// 					int input_col = w_offset + w * stride;
// 					int col_index =
// 						(c * out_height + h) * out_width + w;
// 					float quant_value = im2col_get_pixel(im, height, width,
// 							input_row, input_col, channel, pad) / scale;	// scale
// 					int8_t quantized = (int8_t) round(quant_value);	// round and clamp
// 					col->q[col_index] = quantized;
// 				}
// 			}
// 		}
// 	}
// }

static void maxpool(float *xout, float *x, int height, int width, int nchannels, int ksize, int stride, int pad)
{
    int out_height = (height + 2 * pad - ksize) / stride + 1;
    int out_width = (width + 2 * pad - ksize) / stride + 1;

    for (int c = 0; c < nchannels; c++) {
        int xout_idx = c * out_height * out_width; // start index for xout
        for (int i = 0; i < out_height; i++) {
            for (int j = 0; j < out_width; j++) {
                float cmax = 0;
                for (int ki = 0; ki < ksize; ki++) {
                    for (int kj = 0; kj < ksize; kj++) {
                        int input_row = i * stride + ki - pad;
                        int input_col = j * stride + kj - pad;
                        if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                            cmax = fmax(cmax, x[c * height * width + input_row * width + input_col]);
                        }
                    }
                }
                xout[xout_idx + i * out_width + j] = cmax;
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
// sanity check function for writing tensors,
// e.g., it can be used to evaluate values after a specific layer.
static void write_tensor(float *x, int size)
{
	FILE *f = fopen("tensor.txt", "w");
	for (int i = 0; i < size; i++)
		fprintf(f, "%f\n", x[i]);
	fclose(f);
}

static void write_qtensor(int8_t *x, int size)
{
	FILE *f = fopen("tensorq.txt", "w");
	for (int i = 0; i < size; i++)
		fprintf(f, "%d\n", x[i]);
	fclose(f);
}

static void dequantize(QuantizedTensor *qx, float* x, int n, int gs) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / gs];
    }
}

static void dequantize2d(QuantizedTensor *qx, float* x, int nrows, int ncols, int gs) {
    for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
				x[i * ncols + j] = qx->q[i * ncols + j] * qx->s[(j * nrows + i) / gs];
		}
    }
}
#endif

static void forward(Model * m, uint8_t * image)
{
	ConvConfig *cl = m->conv_config;
	LinearConfig *ll = m->linear_config;
	int8_t *p = m->parameters;
	float *sf = m->scaling_factors;
	Runstate *s = &m->state;
	float *x = s->x;
	float *x2 = s->x2;
	QuantizedTensor xq = s->xq;

	normalize(x2, image);
	im2col_cpu(x, x2, cl[0].ic, 28, 28, cl[0].ksize, cl[0].stride,
		   cl[0].pad);
	quantize2d(&xq, x, cl[0].ksize * cl[0].ksize * cl[0].ic, 28 * 28, cl[0].gs_weight);
	matmul_conv_with_relu(x, &xq, p + cl[0].qoffset, sf + cl[0].soffset, cl[0].oc,
			      cl[0].ic * cl[0].ksize * cl[0].ksize, 28 * 28, cl[0].gs_weight, cl[0].gs_bias);
	maxpool(x2, x, 28, 28, cl[0].oc, 2, 2, 0);
	im2col_cpu(x, x2, cl[1].ic, 14, 14, cl[1].ksize, cl[1].stride,
		   cl[1].pad);
	quantize2d(&xq, x, cl[1].ksize * cl[1].ksize * cl[1].ic, 14 * 14, cl[1].gs_weight);
	matmul_conv_with_relu(x, &xq, p + cl[1].qoffset, sf + cl[1].soffset, cl[1].oc,
			      cl[1].ic * cl[1].ksize * cl[1].ksize, 14 * 14, cl[1].gs_weight, cl[1].gs_bias);
	maxpool(x2, x, 14, 14, cl[1].oc, 2, 2, 0);
	quantize(&xq, x2, 7 * 7 * cl[1].oc, ll[0].gs_weight);
	linear_with_relu(x, &xq, p + ll[0].qoffset, sf + ll[0].soffset, ll[0].in, ll[0].out, ll[0].gs_weight, ll[0].gs_bias);
	quantize(&xq, x, ll[0].out, ll[1].gs_weight);
	linear(x, &xq, p + ll[1].qoffset, sf + ll[1].soffset, ll[1].in, ll[1].out, ll[1].gs_weight, ll[1].gs_bias);
	softmax(x, ll[1].out);
}

static void error_usage()
{
	fprintf(stderr, "Usage:   run <model> <image>\n");
	fprintf(stderr, "Example: run modelq8.bin image1 image2 ... imageN\n");
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
		forward(&model, image);	// output (nclass,) is stored in model.state.x2
		for (int j = 0; j < model.model_config.nclasses; j++) {
			printf("%f\t", model.state.x[j]);
		}
		printf("\n");
	}

	free(image);
	free_model(&model);
	return 0;
}
