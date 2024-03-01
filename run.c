
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
    int n_bn; // the number of batchnorm layers
    int n_linear; // the number of linear layers
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
	int ic;			// input channels
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
} BnConfig;

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int offset;		// the position of the layer parameters in the "parameters" array within the "Model" struct
} LinearConfig;

typedef struct {
    float* x; // buffer to store the input (28*28,)
    float* x2; // buffer to store the output of a layer (25*28*28,)
    float* x3; // buffer to store the output of a layer (9*28*28,)
    float* x4; // buffer to store the output of a layer (9*14*14,)
    float* x5; // buffer to store the output of a layer (9*7*7,)
} Runstate;

typedef struct {
	ModelConfig model_config;
	LinearConfig *linear_config;	// linear layers' config
	ConvConfig *conv_config;	// convolutional layers' config
  BnConfig *bn_config;	// linear layers' config
	float *parameters;	// array of all weigths and biases
	Runstate state;		// the current state in the forward pass
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

void malloc_run_state(Runstate* s) {
    s->x = calloc(256*28*28, sizeof(float));
    s->x2 = calloc(256*28*28, sizeof(float));
    s->x3 = calloc(256*28*28, sizeof(float));
    s->x4 = calloc(256*28*28, sizeof(float));
    s->x5 = calloc(256*28*28, sizeof(float));
}

void free_run_state(Runstate* s) {
    free(s->x);
    free(s->x2);
    free(s->x3);
    free(s->x4);
    free(s->x5);
}

void read_checkpoint(char* path, ModelConfig* config, ConvConfig **convconfig, BnConfig **bnconfig,
                     LinearConfig **linearconfig, float **parameters, int* fd, float** data, size_t* file_size) {
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
    *convconfig = (ConvConfig*) (*data + sizeof(ModelConfig)/sizeof(float));
    *bnconfig = (BnConfig*) (*data + (config->n_conv*sizeof(ConvConfig) + sizeof(ModelConfig))/sizeof(float));
    *linearconfig = (LinearConfig*) (*data + (config->n_conv*sizeof(ConvConfig) + config->n_bn*sizeof(BnConfig) + sizeof(ModelConfig))/sizeof(float));
    int header_size = sizeof(ModelConfig) + config->n_conv * sizeof(ConvConfig) + config->n_bn*sizeof(BnConfig) + config->n_linear * sizeof(LinearConfig);
    *parameters = *data + header_size/sizeof(float); // position the pointer to the start of the parameter data
    
}

void build_model(Model* m, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &m->model_config, &m->conv_config, &m->bn_config, &m->linear_config, &m->parameters, &m->fd, &m->data, &m->file_size);
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

static void linear(float *xout, float *x, float *p, int in, int out, bool relu)
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
    xout[i] = (relu) ? fmax(val + b[i], 0.0f) : val+b[i];
	}
}

static void matmul_conv_with_relu(float *xout, float *x, float *p, int nchannels,
			   int in, int out, bool bias, bool relu)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(chan,) -> xout (nchannels,out)
	int c;
	float *w = p;
	//#pragma omp parallel for private(c)
	for (c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val = 0.0f;
			for (int j = 0; j < in; j++) {
				val += w[c * in + j] * x[j * out + i];
			}
      // float bias_val = (bias) ? b[c] : 0.0f;
      if (bias) {
        float* b = p + nchannels * in;
			  xout[c * out + i] = fmax(val + b[c], 0.0f);
      } else {
        xout[c * out + i] = fmax(val, 0.0f);
      }
		}
	}
}

static void matmul_conv(float *xout, float *x, float *p, int nchannels,
			   int in, int out, bool bias)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(chan,) -> xout (nchannels,out)
	int c;
	float *w = p;
	//#pragma omp parallel for private(c)
	for (c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val = 0.0f;
			for (int j = 0; j < in; j++) {
				val += w[c * in + j] * x[j * out + i];
			}
      if (bias) {
        float *b = p + nchannels * in;
			  xout[c * out + i] = val + b[c];
      } else {
        xout[c * out + i] = val;
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

// sanity check function for writing tensors, e.g., it can be used to evaluate values after a specific layer.
void write_tensor(float* x, int size) {
    FILE* f = fopen("actc.txt", "w");
    for (int i = 0; i < size; i++)
        fprintf(f, "%f\n", x[i]);
    fclose(f);
}

void forward(Model* m, uint8_t* image) {
    ConvConfig *convconfig = m->conv_config;
    BnConfig *bnconfig = m->bn_config;
	  LinearConfig *linearconfig = m->linear_config;
	  float *p = m->parameters;
	  Runstate *s = &m->state;
	  float *x = s->x;
	  float *x2 = s->x2;
    float *x3 = s->x3;
    float *x4 = s->x4;
    float *x5 = s->x5;

    ConvConfig conv0 = convconfig[0];
    ConvConfig conv1 = convconfig[1];
    ConvConfig conv2 = convconfig[2];
    ConvConfig conv3 = convconfig[3];
    ConvConfig conv4 = convconfig[4];
    ConvConfig conv5 = convconfig[5];
    ConvConfig conv6 = convconfig[6];
    ConvConfig conv7 = convconfig[7];
    ConvConfig conv8 = convconfig[8];
    ConvConfig conv9 = convconfig[9];
    ConvConfig conv10 = convconfig[10];
    ConvConfig conv11 = convconfig[11];
    ConvConfig conv12 = convconfig[12];
    ConvConfig conv13 = convconfig[13];
    ConvConfig conv14 = convconfig[14];


    normalize(x, image);
    // (64, 1, 5, 5)

    int wh_in = 28;
    // calculate the size of output
    int wh_out = (wh_in + 2 * conv0.pad - conv0.ksize) / conv0.stride + 1;

    im2col_cpu(x2, x, conv0.ic, 28, 28, conv0.ksize, conv0.stride, conv0.pad);
	  matmul_conv(x, x2, p + conv0.offset, conv0.oc, conv0.ic * conv0.ksize * conv0.ksize, 28 * 28, false);
    //batchnorm_with_relu(x2, x, p + bnconfig[0].offset, bnconfig[0].ic, 28*28);
    batchnorm(x2, x, p + bnconfig[0].offset, bnconfig[0].ic, 28*28, true);
    maxpool(x, x2, 28, 28, conv0.oc, 3, 2, 1);
    matcopy(x2, x, conv0.oc*14*14);
   
    //write_tensor(p + bnconfig[3].offset, 64);
    
    // after maxpool, the size of input is 14*14
    wh_in = 14;
    wh_out = (wh_in + 2 * conv1.pad - conv1.ksize) / conv1.stride + 1;
    // block 1
	  im2col_cpu(x3, x, conv1.ic, 14, 14, conv1.ksize, conv1.stride, conv1.pad);
	  matmul_conv(x, x3, p + conv1.offset, conv1.oc, conv1.ic * conv1.ksize * conv1.ksize, 14 * 14, false);
	  batchnorm(x3, x, p + bnconfig[1].offset, bnconfig[1].ic, 14*14, true);
    im2col_cpu(x, x3, conv2.ic, 14, 14, conv2.ksize, conv2.stride, conv2.pad);
    matmul_conv(x3, x, p + conv2.offset, conv2.oc, conv2.ic * conv2.ksize * conv2.ksize, 14 * 14, false);
    batchnorm(x, x3, p + bnconfig[2].offset, bnconfig[2].ic, 14*14, false);
    // skip connection, no change
    matadd(x, x2, conv2.oc*14*14);
    relu(x, conv2.oc*14*14);
    matcopy(x2, x, conv2.oc*14*14);
    

    // block 2
    im2col_cpu(x3, x, conv3.ic, 14, 14, conv3.ksize, conv3.stride, conv3.pad);
    matmul_conv(x, x3, p + conv3.offset, conv3.oc, conv3.ic * conv3.ksize * conv3.ksize, 14 * 14, false);
    batchnorm(x3, x, p + bnconfig[3].offset, bnconfig[3].ic, 14*14, true);
    im2col_cpu(x, x3, conv4.ic, 14, 14, conv4.ksize, conv4.stride, conv4.pad);
    matmul_conv(x3, x, p + conv4.offset, conv4.oc, conv4.ic * conv4.ksize * conv4.ksize, 14 * 14, false);
    batchnorm(x, x3, p + bnconfig[4].offset, bnconfig[4].ic, 14*14, false);
    // skip connection, no change
    matadd(x, x2, conv4.oc*14*14);
    relu(x, conv4.oc*14*14);
    matcopy(x2, x, conv4.oc*14*14);
     
    //write_tensor(x, conv2.oc*14*14);

    // wh_out = (wh_in + 2 * conv5.pad - conv5.ksize) / conv5.stride + 1;
    // block 3
    im2col_cpu(x4, x, conv5.ic, 14, 14, conv5.ksize, conv5.stride, conv5.pad);
    matmul_conv(x, x4, p + conv5.offset, conv5.oc, conv5.ic * conv5.ksize * conv5.ksize, 7 * 7, false);
    batchnorm(x4, x, p + bnconfig[5].offset, bnconfig[5].ic, 7*7, true);
    im2col_cpu(x, x4, conv6.ic, 7, 7, conv6.ksize, conv6.stride, conv6.pad);
    matmul_conv(x4, x, p + conv6.offset, conv6.oc, conv6.ic * conv6.ksize * conv6.ksize, 7 * 7, false);
    batchnorm(x, x4, p + bnconfig[6].offset, bnconfig[6].ic, 7*7, false);
    // skip connection, change in stride
    im2col_cpu(x4, x2, conv7.ic, 14, 14, conv7.ksize, conv7.stride, conv7.pad);
    matmul_conv(x2, x4, p + conv7.offset, conv7.oc, conv7.ic * conv7.ksize * conv7.ksize, 7 * 7, false);
    batchnorm(x4, x2, p + bnconfig[7].offset, bnconfig[7].ic, 7*7, false);
    matadd(x, x4, conv7.oc*7*7);
    relu(x, conv7.oc*7*7);
    matcopy(x2, x, conv7.oc*7*7);

    // block 4
    im2col_cpu(x4, x, conv8.ic, 7, 7, conv8.ksize, conv8.stride, conv8.pad);
    matmul_conv(x, x4, p + conv8.offset, conv8.oc, conv8.ic * conv8.ksize * conv8.ksize, 7 * 7, false);
    batchnorm(x4, x, p + bnconfig[8].offset, bnconfig[8].ic, 7*7, true);
    im2col_cpu(x, x4, conv9.ic, 7, 7, conv9.ksize, conv9.stride, conv9.pad);
    matmul_conv(x4, x, p + conv9.offset, conv9.oc, conv9.ic * conv9.ksize * conv9.ksize, 7 * 7, false);
    batchnorm(x, x4, p + bnconfig[9].offset, bnconfig[9].ic, 7*7, false);
    // skip connection, no change
    matadd(x, x2, conv9.oc*7*7);
    relu(x, conv9.oc*7*7);
    matcopy(x2, x, conv9.oc*7*7);


    // block 5
    im2col_cpu(x5, x, conv10.ic, 7, 7, conv10.ksize, conv10.stride, conv10.pad);
    matmul_conv(x, x5, p + conv10.offset, conv10.oc, conv10.ic * conv10.ksize * conv10.ksize, 4 * 4, false);
    batchnorm(x5, x, p + bnconfig[10].offset, bnconfig[10].ic, 4*4, true);
    im2col_cpu(x, x5, conv11.ic, 4, 4, conv11.ksize, conv11.stride, conv11.pad);
    matmul_conv(x5, x, p + conv11.offset, conv11.oc, conv11.ic * conv11.ksize * conv11.ksize, 4 * 4, false);
    batchnorm(x, x5, p + bnconfig[11].offset, bnconfig[11].ic, 4*4, false);
    // skip connection, change in stride
    im2col_cpu(x5, x2, conv12.ic, 7, 7, conv12.ksize, conv12.stride, conv12.pad);
    matmul_conv(x2, x5, p + conv12.offset, conv12.oc, conv12.ic * conv12.ksize * conv12.ksize, 4 * 4, false);
    batchnorm(x5, x2, p + bnconfig[12].offset, bnconfig[12].ic, 4*4, false);
    matadd(x, x5, conv12.oc*4*4);
    relu(x, conv12.oc*4*4);
    matcopy(x2, x, conv12.oc*4*4);

    // block 6
    im2col_cpu(x5, x, conv13.ic, 4, 4, conv13.ksize, conv13.stride, conv13.pad);
    matmul_conv(x, x5, p + conv13.offset, conv13.oc, conv13.ic * conv13.ksize * conv13.ksize, 4 * 4, false);
    batchnorm(x5, x, p + bnconfig[13].offset, bnconfig[13].ic, 4*4, true);
    im2col_cpu(x, x5, conv14.ic, 4, 4, conv14.ksize, conv14.stride, conv14.pad);
    matmul_conv(x5, x, p + conv14.offset, conv14.oc, conv14.ic * conv14.ksize * conv14.ksize, 4 * 4, false);
    batchnorm(x, x5, p + bnconfig[14].offset, bnconfig[14].ic, 4*4, false);
    // skip connection, no change
    matadd(x, x2, conv14.oc*4*4);
    relu(x, conv14.oc*4*4);
   

    // global average pooling
    avgpool(x2, x, 4, 4, conv14.oc, 4, 1, 0);
    // linear layer
    linear(x, x2, p + linearconfig[0].offset, linearconfig[0].in, linearconfig[0].out, false);
    softmax(x, linearconfig[0].out);

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
