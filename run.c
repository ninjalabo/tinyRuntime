
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

#define IMAGE_SZ (28*28)

// avoid division by zero
#define eps 0.00001f

typedef struct {
    int nclass; // the number of classes
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
    float* x4; // buffer to store the output of a layer (9*14*14,)
    float* x5; // buffer to store the output of a layer (9*7*7,)
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
    s->x = calloc(64*9*28*28, sizeof(float));
    s->x2 = calloc(64*25*28*28, sizeof(float));
    s->x3 = calloc(64*9*28*28, sizeof(float));
    s->x4 = calloc(128*9*14*14, sizeof(float));
    s->x5 = calloc(256*9*7*7, sizeof(float));
}

void free_run_state(Runstate* s) {
    free(s->x);
    free(s->x2);
    free(s->x3);
    free(s->x4);
    free(s->x5);
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

static void im2col_cpu(float *col, float *im, int height, int width, ConvConfig cc)
{
	// im (nchannels, height, width) -> col (col_size, out_height * out_width)
    int nchannels = cc.ic;
    int ksize = cc.ksize;
    int stride = cc.stride;
    int pad = cc.pad;

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

static void avgpool(float *xout, float *x, int height, int width, int nchannels, int ksize, int stride, int pad){
  int out_height = (height + 2 * pad - ksize) / stride + 1;
  int out_width = (width + 2 * pad - ksize) / stride + 1;
  
  for (int c = 0; c < nchannels; c++) {
        int xout_idx = c * out_height * out_width; // start index for xout
        for (int i = 0; i < out_height; i++) {
            for (int j = 0; j < out_width; j++) {
              float sum = 0.0f;
              for (int ki = 0; ki < ksize; ki++) {
                    for (int kj = 0; kj < ksize; kj++) {
                        int input_row = i * stride + ki - pad;
                        int input_col = j * stride + kj - pad;
                        if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                            sum += x[c * height * width + input_row * width + input_col];
                        }
                    }
                }
                xout[xout_idx + i * out_width + j] = sum / (ksize * ksize);
            }
        }
  }
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
        xout[i] = ((float) image[i] / 255 - 0.5) / 0.5;
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

void read_mnist_image(char* path, uint8_t *image) {
    FILE* file = fopen(path, "rb");
    if (!file) { perror("Error opening file"); exit(EXIT_FAILURE); }
    fread(image, sizeof(uint8_t), IMAGE_SZ, file);
    fclose(file);
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

void forward(Model* m, uint8_t* image) {
    ConvConfig *conv_config = m->conv_config;
	LinearConfig *linear_config = m->linear_config;
    BnConfig *bn_config = m->bn_config;
    float *p = m->parameters;
	Runstate *s = &m->state;
	float *x = s->x;
	float *x2 = s->x2;
    float *x3 = s->x3;
    float *x4 = s->x4;
    float *x5 = s->x5;

    ConvConfig conv0 = conv_config[0];
    ConvConfig conv1 = conv_config[1];
    ConvConfig conv2 = conv_config[2];
    ConvConfig conv3 = conv_config[3];
    ConvConfig conv4 = conv_config[4];
    ConvConfig conv5 = conv_config[5];
    ConvConfig conv6 = conv_config[6];
    ConvConfig conv7 = conv_config[7];
    ConvConfig conv8 = conv_config[8];
    ConvConfig conv9 = conv_config[9];
    ConvConfig conv10 = conv_config[10];
    ConvConfig conv11 = conv_config[11];
    ConvConfig conv12 = conv_config[12];
    ConvConfig conv13 = conv_config[13];
    ConvConfig conv14 = conv_config[14];

    normalize(x, image);

    im2col_cpu(x2, x, 28, 28, conv0);
	matmul_conv(x, x2, p, conv0, 28 * 28, false, false);
    //batchnorm_with_relu(x2, x, p + bnconfig[0].offset, bnconfig[0].ic, 28*28);
    batchnorm(x2, x, p + bn_config[0].offset, bn_config[0].ic, 28*28, true);
    maxpool(x, x2, 28, 28, conv0.oc, 3, 2, 1);
    matcopy(x2, x, conv0.oc*14*14);

    // block 1
	im2col_cpu(x3, x, 14, 14, conv1);
	matmul_conv(x, x3, p, conv1, 14 * 14, false, false);
	batchnorm(x3, x, p + bn_config[1].offset, bn_config[1].ic, 14*14, true);
    im2col_cpu(x, x3, 14, 14, conv2);
    matmul_conv(x3, x, p, conv2, 14 * 14, false, false);
    batchnorm(x, x3, p + bn_config[2].offset, bn_config[2].ic, 14*14, false);
    // skip connection, no change
    matadd(x, x2, conv2.oc*14*14);
    relu(x, conv2.oc*14*14);
    matcopy(x2, x, conv2.oc*14*14);

    // block 2
    im2col_cpu(x3, x, 14, 14, conv3);
    matmul_conv(x, x3, p, conv3, 14 * 14, false, false);
    batchnorm(x3, x, p + bn_config[3].offset, bn_config[3].ic, 14*14, true);
    im2col_cpu(x, x3, 14, 14, conv4);
    matmul_conv(x3, x, p, conv4, 14 * 14, false, false);
    batchnorm(x, x3, p + bn_config[4].offset, bn_config[4].ic, 14*14, false);
    // skip connection, no change
    matadd(x, x2, conv4.oc*14*14);
    relu(x, conv4.oc*14*14);
    matcopy(x2, x, conv4.oc*14*14);

    // block 3
    im2col_cpu(x4, x, 14, 14, conv5);
    matmul_conv(x, x4, p, conv5, 7 * 7, false, false);
    batchnorm(x4, x, p + bn_config[5].offset, bn_config[5].ic, 7*7, true);
    im2col_cpu(x, x4, 7, 7, conv6);
    matmul_conv(x4, x, p, conv6, 7 * 7, false, false);
    batchnorm(x, x4, p + bn_config[6].offset, bn_config[6].ic, 7*7, false);

    // skip connection, change in stride
    im2col_cpu(x4, x2, 14, 14, conv7);
    matmul_conv(x2, x4, p, conv7, 7 * 7, false, false);
    batchnorm(x4, x2, p + bn_config[7].offset, bn_config[7].ic, 7*7, false);
    matadd(x, x4, conv7.oc*7*7);
    relu(x, conv7.oc*7*7);
    matcopy(x2, x, conv7.oc*7*7);

    // block 4
    im2col_cpu(x4, x, 7, 7, conv8);
    matmul_conv(x, x4, p, conv8, 7 * 7, false, false);
    batchnorm(x4, x, p + bn_config[8].offset, bn_config[8].ic, 7*7, true);
    im2col_cpu(x, x4, 7, 7, conv9);
    matmul_conv(x4, x, p, conv9, 7 * 7, false, false);
    batchnorm(x, x4, p + bn_config[9].offset, bn_config[9].ic, 7*7, false);
    // skip connection, no change
    matadd(x, x2, conv9.oc*7*7);
    relu(x, conv9.oc*7*7);
    matcopy(x2, x, conv9.oc*7*7);

    // block 5
    im2col_cpu(x5, x, 7, 7, conv10);
    matmul_conv(x, x5, p, conv10, 4 * 4, false, false);
    batchnorm(x5, x, p + bn_config[10].offset, bn_config[10].ic, 4*4, true);
    im2col_cpu(x, x5, 4, 4, conv11);
    matmul_conv(x5, x, p, conv11, 4 * 4, false, false);
    batchnorm(x, x5, p + bn_config[11].offset, bn_config[11].ic, 4*4, false);
    // skip connection, change in stride
    im2col_cpu(x5, x2, 7, 7, conv12);
    matmul_conv(x2, x5, p, conv12, 4 * 4, false, false);
    batchnorm(x5, x2, p + bn_config[12].offset, bn_config[12].ic, 4*4, false);
    matadd(x, x5, conv12.oc*4*4);
    relu(x, conv12.oc*4*4);
    matcopy(x2, x, conv12.oc*4*4);

    // block 6
    im2col_cpu(x5, x, 4, 4, conv13);
    matmul_conv(x, x5, p, conv13, 4 * 4, false, false);
    batchnorm(x5, x, p + bn_config[13].offset, bn_config[13].ic, 4*4, true);
    im2col_cpu(x, x5, 4, 4, conv14);
    matmul_conv(x5, x, p, conv14, 4 * 4, false, false);
    batchnorm(x, x5, p + bn_config[14].offset, bn_config[14].ic, 4*4, false);
    // skip connection, no change
    matadd(x, x2, conv14.oc*4*4);
    relu(x, conv14.oc*4*4);

    // global average pooling
    avgpool(x2, x, 4, 4, conv14.oc, 4, 1, 0);
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
    char* model_path = NULL;
    char* image_path = NULL;

    // read images and model path, then outputs the probability distribution for the given images.
    if (argc < 3) { error_usage(); }
    model_path = argv[1]; 
    Model model;
    build_model(&model, model_path);
    uint8_t* image = malloc(IMAGE_SZ);
    for (int i = 2; i < argc; i++) {
        image_path = argv[i];
        read_mnist_image(image_path, image);
        forward(&model, image); // output (nclass,) is stored in model.state.x
        for (int j = 0; j < model.model_config.nclass; j++) {
             printf("%f\t", model.state.x[j]);
         }
        printf("\n");
    }

    free(image);
    free_model(&model);
    return 0;
}
