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
#include <stdbool.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define IMAGE_SZ (3 * 224 * 224)        // model input image size (all images are resized to this)
#define MAX_IMAGE_SZ 562500         // max image size in Imagenette

// avoid division by zero
#define eps 0.00001f

typedef struct {
	int8_t *q;		// quantized values
	float *s;		// scaling factors
} QuantizedTensor;

typedef struct {
	int nclasses;		// the number of classes
	int nconv;		// the number of convolutional layers
	int nlinear;		// the number of linear layers
	int nbn;		// the number of batchnorm layers
	int nparameters;		// the number of parameters
} ModelConfig;

typedef struct {
	int ksize;		// kernel size
	int stride;
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int qoffset;		// offset for quantized parameters
	int soffset;		// offset for scaling factors
	int gs_weight;			// group size for weights
} ConvConfig;

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int qoffset;		// offset for quantized parameters
	int soffset;		// offset for scaling factors
	int gs_weight;		// group size for weights
	int gs_bias;		// group size for biases
} LinearConfig;

typedef struct {
	int ic;			// input channels
	int offset;		// offset for parameters (stored in scaling_factors as parameters are non-quantized)
} BnConfig;

typedef struct {
	float *x;		// buffer to store the output of a layer (IMAGE_SZ,)
	float *x2;
	float *x3;
	float *x4;
	float *x5;
	QuantizedTensor xq;	// buffer for quantized arrays
} Runstate;

typedef struct {
	ModelConfig model_config;
	ConvConfig *conv_config;	// convolutional layers' config
	LinearConfig *linear_config;	// linear layers' config
	BnConfig *bn_config;		// batchnorm layers' config
	int8_t *parameters;	// array of all weigths and biases
	float *scaling_factors; // array of all scaling factors
	Runstate state;		// the current state in the forward pass
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

static void malloc_run_state(Runstate *s)
{
    s->x = calloc(64*9*56*56, sizeof(float));
    s->x2 = calloc(3*49*112*112, sizeof(float));
    s->x3 = calloc(64*9*56*56, sizeof(float));
	s->xq = (QuantizedTensor) {
	.q = calloc(3*49*112*112, sizeof(int8_t)),.s =
		    calloc(64*56*56, sizeof(float))};
}

static void free_run_state(Runstate *s)
{
	free(s->x);
    free(s->x2);
    free(s->x3);
	free(s->xq.q);
	free(s->xq.s);
}

static void read_checkpoint(char *path, ModelConfig * config, ConvConfig ** cc,
		     LinearConfig ** lc, BnConfig ** bc, int8_t **parameters,
			 float **scaling_factors, int *fd, float **data, size_t *file_size)
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
	*cc = (ConvConfig *) (*data + sizeof(ModelConfig) / sizeof(float));
	*lc =
	    (LinearConfig *) (*data +
			      (config->nconv * sizeof(ConvConfig) +
			       sizeof(ModelConfig)) / sizeof(float));
	*bc =
		(BnConfig*) (*data + 
				  (config->nconv*sizeof(ConvConfig) +
				   config->nlinear * sizeof(LinearConfig) +
		           sizeof(ModelConfig)) / sizeof(float));
	// memory map weights and biases
	int header_size =
	    sizeof(ModelConfig) + config->nconv * sizeof(ConvConfig) +
	    config->nlinear * sizeof(LinearConfig) +
		config->nbn * sizeof(BnConfig);
	*parameters = (int8_t *) *data + header_size;	// position the parameters pointer to the start of the parameter data
	*scaling_factors = (float*) (*parameters + config->nparameters);
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

static void quantize2d(QuantizedTensor *qx, float *x, ConvConfig cc, int ncols)
{
	int nrows = cc.ic * cc.ksize * cc.ksize;
	int gs = cc.gs_weight;
	// quantize each column in 2d tensor with a specified group size
	int num_groups = nrows / gs;
	float Q_MAX = 127.0f;

	for (int col = 0; col < ncols; col++) {
		for (int group = 0; group < num_groups; group++) {
			// find the max absolute value in the current group
			float wmax = 0.0;
			for (int i = 0; i < gs; i++) {
				float val = fabs(x[(group * gs + i) * ncols + col]);
				if (val > wmax) {
					wmax = val;
				}
			}
			// calculate and write the scaling factor
			float scale = wmax / Q_MAX;
			qx->s[col * num_groups + group] = scale;

			// calculate and write the quantized values
			for (int i = 0; i < gs; i++) {
				float quant_value = x[(group * gs + i) * ncols + col] / scale;	// scale
				int8_t quantized = (int8_t) round(quant_value);	// round and clamp
				qx->q[(group * gs + i) * ncols + col] = quantized;
			}
		}
	}
}

static void linear(float *xout, QuantizedTensor *x, int8_t *p,
			float *sf, LinearConfig lc, bool relu)
{
	// linear layer: w(out,in) @ x (in,) + b(out,) -> xout (out,)
	// by far the most amount of time is spent inside this little function
	int in = lc.in;
	int out = lc.out;
	int gs_w = lc.gs_weight;
	int gs_b = lc.gs_bias;

	int i;
	int8_t *w = p + lc.qoffset;
	int8_t *b = w + in * out;
	float *sw = sf + lc.soffset;
	float *sb = sw + in * out / gs_w;
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
		xout[i] = relu ? fmax(0.0f, val + ((float) b[i]) * sb[i / gs_b]) : 
						 (val + ((float) b[i]) * sb[i / gs_b]);
	}
}

// linear and convolutional layer operations work only if gs <= in and in % gs = 0
static void matmul_conv(float *xout, QuantizedTensor *x, int8_t *p,
				float *sf, ConvConfig cc, int out, bool relu)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(chan,) -> xout (nchannels,out)
	int nchannels = cc.oc;
    int in = cc.ic * cc.ksize * cc.ksize;
	int gs_w = cc.gs_weight;

	int c;
	int8_t *w = p + cc.qoffset;
	float *sw = sf + cc.soffset;
	//#pragma omp parallel for private(c)
	for (c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val = 0.0f;
			// do the matmul in groups of gs_w
			for (int j = 0; j < in; j += gs_w) {
				int32_t ival = 0;
				for (int k = 0; k < gs_w; k++) {
					ival += ((int32_t) x->q[(j + k) * out + i]) *
					((int32_t) w[c * in + (j + k)]);
				}
				val += (float) ival * x->s[(i * in + j) / gs_w] * sw[(c * in + j) / gs_w];
			}
			xout[c * out + i] = relu ? fmax(0, val) : val;
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

static void im2col_cpu(float *col, float *im, int *height, int *width, ConvConfig cc)
{
	// im (nchannels, height, width) -> col (col_size, out_height * out_width)
    int nchannels = cc.ic;
    int ksize = cc.ksize;
    int stride = cc.stride;
    int pad = cc.pad;
	
	int c, h, w;
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;

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
				    im2col_get_pixel(im, *height, *width,
						     input_row, input_col,
						     channel, pad);
			}
		}
	}
	// update current height and width
    *height = out_height;
    *width = out_width;
}

static void batchnorm(float *xout, float *x, float *p, int nchannels, int in, bool relu){
    // x (nchannels,in) -> xout (nchannels,in)
    float *w = p;
    float *b = p + nchannels;
    float *running_mean = p + 2 * nchannels;
    float *running_var = p + 3 * nchannels;
    for (int c = 0; c < nchannels; c++){
      for (int i = 0; i < in; i++){
        float val = (x[c * in + i] - running_mean[c]) / sqrt(running_var[c] + eps) * w[c] + b[c];
        xout[c * in + i] = (relu) ? fmax(val, 0.0f) : val;
      }
    }
}

static void maxpool(float *xout, float *x, int *height, int *width, int nchannels, int ksize, int stride, int pad)
{
    int out_height = (*height + 2 * pad - ksize) / stride + 1;
    int out_width = (*width + 2 * pad - ksize) / stride + 1;

    for (int c = 0; c < nchannels; c++) {
        int xout_idx = c * out_height * out_width; // start index for xout
        for (int i = 0; i < out_height; i++) {
            for (int j = 0; j < out_width; j++) {
                float cmax = 0;
                for (int ki = 0; ki < ksize; ki++) {
                    for (int kj = 0; kj < ksize; kj++) {
                        int input_row = i * stride + ki - pad;
                        int input_col = j * stride + kj - pad;
                        if (input_row >= 0 && input_row < *height && input_col >= 0 && input_col < *width) {
                            cmax = fmax(cmax, x[c * (*height) * (*width) + input_row * (*width) + input_col]);
                        }
                    }
                }
                xout[xout_idx + i * out_width + j] = cmax;
            }
        }
    }
    *height = out_height;
    *width = out_width;
}

static void avgpool(float *xout, float *x, int *height, int *width, int nchannels, int ksize, int stride, int pad){
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;
  
	for (int c = 0; c < nchannels; c++) {
        int xout_idx = c * out_height * out_width; // start index for xout
        for (int i = 0; i < out_height; i++) {
            for (int j = 0; j < out_width; j++) {
              float sum = 0.0f;
              for (int ki = 0; ki < ksize; ki++) {
                    for (int kj = 0; kj < ksize; kj++) {
                        int input_row = i * stride + ki - pad;
                        int input_col = j * stride + kj - pad;
                        if (input_row >= 0 && input_row < *height && input_col >= 0 && input_col < *width) {
                            sum += x[c * (*height) * (*width) + input_row * (*width) + input_col];
                        }
                    }
                }
                xout[xout_idx + i * out_width + j] = sum / (ksize * ksize);
            }
        }
	}
    *height = out_height;
    *width = out_width;
}

void matadd(float* x, float* y, int size){
    for (int i = 0; i < size; i++){
        x[i] = x[i] + y[i];
    }
}

void relu(float* x, int size) {
    // apply ReLU (Rectified Linear Unit) activation 
    for (int i = 0; i < size; i++){
        x[i] = fmax(0.0f, x[i]);
    }
}

void matcopy(float* x, float* y, int size) {
    for (int i = 0; i < size; i++){
        x[i] = y[i];
    }
}

static void normalize(float *xout, uint8_t * image)
{
	// normalize values [0, 255] -> [-1, 1]
	for (int i = 0; i < IMAGE_SZ; i++) {
		xout[i] = ((float) image[i] / 255);
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

// FIX: the results is different compared to transforms.Resize(224, 224) in pytorch
void bilinear_interpolation(uint8_t **resized_image, uint8_t *image, int input_height,
                             int input_width) {
    // resize image to 224 x 224 using bilinear interpolation
    int nchannels = 3;
    int out_height = 224;
    int out_width = 224;

    for (int c = 0; c < nchannels; ++c) {
        int im_idx = c * input_height * input_width;        // start index for image
        int rim_idx = c * out_height * out_width;       // start index for resized image
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                // Calculate corresponding position in the source image
                float src_h_f = h * (input_height - 1) / (float)(out_height - 1);
                float src_w_f = w * (input_width - 1) / (float)(out_width - 1);

                int src_h0 = floor(src_h_f);
                int src_w0 = floor(src_w_f);
                int src_h1 = src_h0 + 1;
                int src_w1 = src_w0 + 1;

                // Perform bilinear interpolation
                (*resized_image)[rim_idx + h * out_width + w] =
                    (1 - fabs(src_h_f - src_h0)) * (1 - fabs(src_w_f - src_w0)) * image[im_idx + src_h0 * input_width + src_w0] +
                    (1 - fabs(src_h_f - src_h0)) * (1 - fabs(src_w_f - src_w1)) * image[im_idx + src_h0 * input_width + src_w1] +
                    (1 - fabs(src_h_f - src_h1)) * (1 - fabs(src_w_f - src_w0)) * image[im_idx + src_h1 * input_width + src_w0] +
                    (1 - fabs(src_h_f - src_h1)) * (1 - fabs(src_w_f - src_w1)) * image[im_idx + + src_h1 * input_width + src_w1];
            }
        }
    }
}

// FIX results different compared to transforms.CenterCrop(224, 224) in pytorch
void center_crop(uint8_t** resized_image, uint8_t* image, int height, int width) {
    int target_height = 224;
    int target_width = 224;
    if (target_height > height || target_width > width) {
        printf("Error: Target size is larger than the input image.\n");
        exit(EXIT_FAILURE);
    }
    int start_row = (height - target_height) / 2;
    int start_col = (width - target_width) / 2;

    // copy the cropped region
    for (int c = 0; c < 3; c++) {
        int im_idx = c * height * width;        // start index for image
        int rim_idx = c * target_height * target_width;       // start index for resized image
        for (int i = 0; i < target_height; i++) {
            for (int j = 0; j < target_width; j++) {
                (*resized_image)[rim_idx + i * target_width + j] = image[im_idx + (i + start_row) * width + j + start_col];
            }
        }
    }
}

void read_imagenette_image(char *path, uint8_t **image, int *height, int *width) {
    // read the image and its size using stb_image
    int nchannels;

    uint8_t *data = stbi_load(path, width, height, &nchannels, 0);

    if (!data) {
        fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
        exit(EXIT_FAILURE);
    }
    if (nchannels != 3) {
        fprintf(stderr, "Number of channels doesn't match\n");
        exit(EXIT_FAILURE);
    }

    // Permute dimensions to (C x H x W) format
    for (int c = 0; c < nchannels; ++c) {
        for (int h = 0; h < (*height); ++h) {
            for (int w = 0; w < (*width); ++w) {
                (*image)[c * (*height) * (*width) + h * (*width)+ w] = data[(h * (*width) + w) * nchannels + c];
            }
        }
    }
    free(data);
}

#define DEBUG
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

static void write_qtensor(int8_t *x, int size)
{
	FILE *f = fopen("testq.txt", "w");
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

void forward(Model* m, uint8_t* resized_image) {
    ConvConfig *cc = m->conv_config;
	LinearConfig *lc = m->linear_config;
	BnConfig *bc = m->bn_config;
    int8_t *p = m->parameters;
	float *sf = m->scaling_factors;
	Runstate *s = &m->state;
	float *x = s->x;
	float *x2 = s->x2;
    float *x3 = s->x3;
	QuantizedTensor xq = s->xq;

    int h = 224;        // height
    int w = 224;        // width
    int h_prev;         // buffer to store previous height for skip connection
    int w_prev;         // buffer to store previous width for skip connection

    normalize(x, resized_image);

	im2col_cpu(x2, x, &h, &w, cc[0]);
	quantize2d(&xq, x2, cc[0], h * w);
	matmul_conv(x, &xq, p, sf, cc[0], h * w, false);
    batchnorm(x2, x, sf + bc[0].offset, bc[0].ic, h * w, true);
    maxpool(x, x2, &h, &w, cc[0].oc, 3, 2, 1);
    matcopy(x2, x, cc[0].oc * h * w);

    // block 1.1 and 1.2
    for (int i = 1; i < 4; i += 2) {
        im2col_cpu(x3, x, &h, &w, cc[i]);
		quantize2d(&xq, x3, cc[i], h * w);
        matmul_conv(x, &xq, p, sf, cc[i], h * w, false);
        batchnorm(x3, x, sf + bc[i].offset, bc[i].ic, h * w, true);
        im2col_cpu(x, x3, &h, &w, cc[i + 1]);
		quantize2d(&xq, x, cc[i + 1], h * w);
        matmul_conv(x3, &xq, p, sf, cc[i + 1], h * w, false);
        batchnorm(x, x3, sf + bc[i + 1].offset, bc[i + 1].ic, h * w, false);
        // skip connection, no change
        matadd(x, x2, cc[i + 1].oc * h * w);
        relu(x, cc[i + 1].oc * h * w);
        matcopy(x2, x, cc[i + 1].oc * h * w);
    }

    // block 2-4
    for (int i = 5; i < 16; i += 5) {
        // block i.1
        h_prev = h;
        w_prev = w;
        im2col_cpu(x3, x, &h, &w, cc[i]);
		quantize2d(&xq, x3, cc[i], h * w);
        matmul_conv(x, &xq, p, sf, cc[i], h * w, false);
        batchnorm(x3, x, sf + bc[i].offset, bc[i].ic, h * w, true);
        im2col_cpu(x, x3, &h, &w, cc[i + 1]);
		quantize2d(&xq, x, cc[i + 1], h * w);
        matmul_conv(x3, &xq, p, sf, cc[i + 1], h * w, false);
        batchnorm(x, x3, sf + bc[i + 1].offset, bc[i + 1].ic, h * w, false);

        // skip connection, change in stride
        im2col_cpu(x3, x2, &h_prev, &w_prev, cc[i + 2]);
		quantize2d(&xq, x3, cc[i + 2], h * w);
        matmul_conv(x2, &xq, p, sf, cc[i + 2], h * w, false);
        batchnorm(x3, x2, sf + bc[i + 2].offset, bc[i + 2].ic, h * w, false);
        matadd(x, x3, cc[i + 2].oc * h * w);
        relu(x, cc[i + 2].oc * h * w);
        matcopy(x2, x, cc[i + 2].oc * h * w);

        // block i.2
        im2col_cpu(x3, x, &h, &w, cc[i + 3]);
		quantize2d(&xq, x3, cc[i + 3], h * w);
        matmul_conv(x, &xq, p, sf, cc[i + 3], h * w, false);
        batchnorm(x3, x, sf + bc[i + 3].offset, bc[i + 3].ic, h * w, true);
        im2col_cpu(x, x3, &h, &w, cc[i + 4]);
		quantize2d(&xq, x, cc[i + 4], h * w);
        matmul_conv(x3, &xq, p, sf, cc[i + 4], h * w, false);
        batchnorm(x, x3, sf + bc[i + 4].offset, bc[i + 4].ic, h * w, false);
        // skip connection, no change
        matadd(x, x2, cc[i + 4].oc * h * w);
        relu(x, cc[i + 4].oc * h * w);
        // the final block output doesn't need to be copied
        if (i < 11) {
            matcopy(x2, x, cc[i + 4].oc * h * w);
        }
    }

    // global average pooling
    avgpool(x2, x, &h, &w, cc[19].oc, h, 1, 0);
    // linear layer
	quantize(&xq, x2, lc[0].in, lc[0].gs_weight);
    linear(x, &xq, p, sf, lc[0], false);
    softmax(x, lc[0].out);
}

static void error_usage()
{
	fprintf(stderr, "Usage:   run <model> <image>\n");
	fprintf(stderr, "Example: run modelq8.bin image1 image2 ... imageN\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
    char *model_path = NULL;
    char *image_path = NULL;

    // read images and model path, then outputs the probability distribution for the given images.
    if (argc < 3) { error_usage(); }
    model_path = argv[1];
    Model model;
    build_model(&model, model_path);

    int input_height;
    int input_width;
    uint8_t *image = malloc(MAX_IMAGE_SZ);
    uint8_t *resized_image = malloc(IMAGE_SZ);

    for (int i = 2; i < argc; i++) {
        image_path = argv[i];
        // read input image, its height and width
        read_imagenette_image(image_path, &image, &input_height, &input_width);
        // resize image to 224 x 224, bilinear interpolation or center crop
        // bilinear_interpolation(&resized_image, image, input_height, input_width);
        center_crop(&resized_image, image, input_height, input_width);
        forward(&model, resized_image); // output (nclass,) is stored in model.state.x
        for (int j = 0; j < model.model_config.nclasses; j++) {
            printf("%f\t", model.state.x[j]);
        }
        printf("\n");
    }

    free(image);
    free(resized_image);
    free_model(&model);
    return 0;
}
