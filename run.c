
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
    int nclasses; // the number of classes
    int n_conv; // the number of convolutional layers
    int n_linear; // the number of linear layers
    int n_bn; // the number of batchnorm layers
} ModelConfig;

typedef struct {
	int ksize;		// kernel size
	int stride;
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
} ConvConfig;

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
} LinearConfig;

typedef struct {
	int ic;			// input channels
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
} BnConfig;

typedef struct {
    float* x; // buffer to store the input (28*28,)
    float* x2; // buffer to store the output of a layer (25*28*28,)
    float* x3; // buffer to store the output of a layer (9*28*28,)
} Runstate;

typedef struct {
	ModelConfig model_config;
    ConvConfig *conv_config;	// convolutional layers' config
	LinearConfig *linear_config;	// linear layers' config
    BnConfig *bn_config;	// linear layers' config
	float *parameters;	// array of all weigths and biases
	Runstate state;		// the current state in the forward pass
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

void malloc_run_state(Runstate* s) {
    s->x = calloc(64*9*56*56, sizeof(float));
    s->x2 = calloc(3*49*112*112, sizeof(float));
    s->x3 = calloc(64*9*56*56, sizeof(float));
}

void free_run_state(Runstate* s) {
    free(s->x);
    free(s->x2);
    free(s->x3);
}

void read_checkpoint(char* path, ModelConfig* config, ConvConfig **conv_config, LinearConfig **linear_config, 
                     BnConfig** bn_config, float **parameters, int* fd, float** data, size_t* file_size) {
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", path); exit(EXIT_FAILURE); }
    // read model config
    if (fread(config, sizeof(ModelConfig), 1, file) != 1) { exit(EXIT_FAILURE); }
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the model weights into the data pointer
    *fd = open(path, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    *conv_config = (ConvConfig*) (*data + sizeof(ModelConfig)/sizeof(float));
    *linear_config = (LinearConfig*) (*data + (config->n_conv*sizeof(ConvConfig) + sizeof(ModelConfig))/sizeof(float));
    *bn_config = (BnConfig*) (*data + (config->n_conv*sizeof(ConvConfig) + config->n_linear*sizeof(LinearConfig) + sizeof(ModelConfig))/sizeof(float));
    int header_size = sizeof(ModelConfig) + config->n_conv * sizeof(ConvConfig) + config->n_linear * sizeof(LinearConfig) + config->n_bn * sizeof(BnConfig);
    *parameters = *data + header_size/sizeof(float); // position the pointer to the start of the parameter data
    
}

void build_model(Model* m, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &m->model_config, &m->conv_config, &m->linear_config, &m->bn_config, &m->parameters, &m->fd, &m->data, &m->file_size);
    // allocate the RunState buffers
    malloc_run_state(&m->state);
}

void free_model(Model* m) {
    // close the memory mapping
    if (m->data != MAP_FAILED) { munmap(m->data, m->file_size); }
    if (m->fd != -1) { close(m->fd); }
    // free the RunState buffers
    free_run_state(&m->state);
}

static void linear(float *xout, float *x, float *p, LinearConfig lc, bool relu)
{
	// linear layer with ReLU activation: w(out,in) @ x (in,) + b(out,) -> xout (out,)
    int in = lc.in;
    int out = lc.out;

	int i;
	float *w = p + lc.offset;
	float *b = w + in * out;
	//#pragma omp parallel for private(i)
	for (i = 0; i < out; i++) {
		float val = 0.0f;
		for (int j = 0; j < in; j++) {
			val += w[i * in + j] * x[j];
		}
    xout[i] = (relu) ? fmax(val + b[i], 0.0f) : val + b[i];
	}
}

static void matmul_conv(float *xout, float *x, float *p, ConvConfig cc, int out, bool bias, bool relu)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(nchannels,) -> xout (nchannels,out)
    int nchannels = cc.oc;
    int in = cc.ic * cc.ksize * cc.ksize;

	int c;
	float *w = p + cc.offset;
    float *b = w + nchannels * in;
	//#pragma omp parallel for private(c)
	for (c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val = 0.0f;
			for (int j = 0; j < in; j++) {
				val += w[c * in + j] * x[j * out + i];
			}
            float bias_val = (bias) ? b[c] : 0.0f;
			xout[c * out + i] = relu ? fmax(val + bias_val, 0.0f) : (val + bias_val);
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

void normalize(float* xout, uint8_t* image) {
    // normalize values [0, 255] -> [-1, 1]
    for (int i = 0; i < IMAGE_SZ; i++) {
        xout[i] = ((float) image[i] / 255);
    }
}

void softmax(float* x, int size) {
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

#ifdef DEBUG
// sanity check function for writing tensors, e.g., it can be used to evaluate values after a specific layer.
void write_tensor(float* x, int size) {
    FILE* f = fopen("test1.txt", "w");
    for (int i = 0; i < size; i++)
        fprintf(f, "%f\n", x[i]);
    fclose(f);
}
#endif

void forward(Model* m, uint8_t* resized_image) {
    ConvConfig *conv_config = m->conv_config;
	LinearConfig *linear_config = m->linear_config;
    BnConfig *bn_config = m->bn_config;
    float *p = m->parameters;
	Runstate *s = &m->state;
	float *x = s->x;
	float *x2 = s->x2;
    float *x3 = s->x3;

    int h = 224;        // height
    int w = 224;        // width
    int h_prev;         // buffer to store previous height for skip connection
    int w_prev;         // buffer to store previous width for skip connection

    normalize(x, resized_image);

    im2col_cpu(x2, x, &h, &w, conv_config[0]);
	matmul_conv(x, x2, p, conv_config[0], h * w, false, false);
    batchnorm(x2, x, p + bn_config[0].offset, bn_config[0].ic, h * w, true);
    maxpool(x, x2, &h, &w, conv_config[0].oc, 3, 2, 1);
    matcopy(x2, x, conv_config[0].oc * h * w);

    // block 1.1 and 1.2
    for (int i = 1; i < 4; i += 2) {
        im2col_cpu(x3, x, &h, &w, conv_config[i]);
        matmul_conv(x, x3, p, conv_config[i], h * w, false, false);
        batchnorm(x3, x, p + bn_config[i].offset, bn_config[i].ic, h * w, true);
        im2col_cpu(x, x3, &h, &w, conv_config[i + 1]);
        matmul_conv(x3, x, p, conv_config[i + 1], h * w, false, false);
        batchnorm(x, x3, p + bn_config[i + 1].offset, bn_config[i + 1].ic, h * w, false);
        // skip connection, no change
        matadd(x, x2, conv_config[i + 1].oc * h * w);
        relu(x, conv_config[i + 1].oc * h * w);
        matcopy(x2, x, conv_config[i + 1].oc * h * w);
    }

    // block 2-4
    for (int i = 5; i < 16; i += 5) {
        // block i.1
        h_prev = h;
        w_prev = w;
        im2col_cpu(x3, x, &h, &w, conv_config[i]);
        matmul_conv(x, x3, p, conv_config[i], h * w, false, false);
        batchnorm(x3, x, p + bn_config[i].offset, bn_config[i].ic, h * w, true);
        im2col_cpu(x, x3, &h, &w, conv_config[i + 1]);
        matmul_conv(x3, x, p, conv_config[i + 1], h * w, false, false);
        batchnorm(x, x3, p + bn_config[i + 1].offset, bn_config[i + 1].ic, h * w, false);

        // skip connection, change in stride
        im2col_cpu(x3, x2, &h_prev, &w_prev, conv_config[i + 2]);
        matmul_conv(x2, x3, p, conv_config[i + 2], h * w, false, false);
        batchnorm(x3, x2, p + bn_config[i + 2].offset, bn_config[i + 2].ic, h * w, false);
        matadd(x, x3, conv_config[i + 2].oc * h * w);
        relu(x, conv_config[i + 2].oc * h * w);
        matcopy(x2, x, conv_config[i + 2].oc * h * w);

        // block i.2
        im2col_cpu(x3, x, &h, &w, conv_config[i + 3]);
        matmul_conv(x, x3, p, conv_config[i + 3], h * w, false, false);
        batchnorm(x3, x, p + bn_config[i + 3].offset, bn_config[i + 3].ic, h * w, true);
        im2col_cpu(x, x3, &h, &w, conv_config[i + 4]);
        matmul_conv(x3, x, p, conv_config[i + 4], h * w, false, false);
        batchnorm(x, x3, p + bn_config[i + 4].offset, bn_config[i + 4].ic, h * w, false);
        // skip connection, no change
        matadd(x, x2, conv_config[i + 4].oc * h * w);
        relu(x, conv_config[i + 4].oc * h * w);
        // the final block output doesn't need to be copied
        if (i < 11) {
            matcopy(x2, x, conv_config[i + 4].oc * h * w);
        }
    }

    // global average pooling
    avgpool(x2, x, &h, &w, conv_config[19].oc, h, 1, 0);
    // linear layer
    linear(x, x2, p, linear_config[0], false);
    softmax(x, linear_config[0].out);
}

void error_usage() {
    fprintf(stderr, "Usage:   run <model> <image>\n");
    fprintf(stderr, "Example: run model.bin image1 image2 ... imageN\n");
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
