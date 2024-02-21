
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
    int dim; // model dimension
    int nclass; // the number of classes
} Config;

typedef struct {
    float* wc1; // conv1 weights (4, 1, 3, 3)
    float* bc1; // conv1 biases (4,)
    float* wc2; // conv2 weights (8, 4, 3, 3)
    float* bc2; // conv2 biases (8,)
    float* wf1; // fc1 weights (dim, 8*7*7)
    float* bf1; // fc1 biases (dim,)
    float* wf2; // fc2 weights (nclass, dim)
    float* bf2; // fc2 biases (nclass,)
} Weights;

typedef struct {
    float* x; // buffer to store the output of a layer (9*28*28,)
    float* x2; // buffer to store the output of a layer (4*28*28,)
} Runstate;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    Weights weights; // the weights of the model
    Runstate state; // the current state in the forward pass
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    size_t file_size; // size of the checkpoint file in bytes
} Model;

void malloc_run_state(Runstate* s, int dim) {
    s->x = calloc(9*28*28, sizeof(float));
    s->x2 = calloc(4*28*28, sizeof(float));
}

void free_run_state(Runstate* s) {
    free(s->x);
    free(s->x2);
}

void memory_map_weights(Weights* w, Config* c, float* ptr) {
    // maps memory for the weights and bias of each layer
    int dim = c->dim;
    int nclass = c->nclass;
    w->wc1 = ptr;
    ptr += 4*3*3;
    w->bc1 = ptr;
    ptr += 4;
    w->wc2 = ptr;
    ptr +=  8*4*3*3;
    w->bc2 = ptr;
    ptr += 8;
    w->wf1 = ptr;
    ptr += 8*7*7 * dim;
    w->bf1 = ptr;
    ptr += dim;
    w->wf2 = ptr;
    ptr += dim * nclass;
    w->bf2 = ptr;
}

void read_checkpoint(char* path, Config* config, Weights* weights, int* fd, float** data, size_t* file_size) {
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", path); exit(EXIT_FAILURE); }
    // read model config
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the model weights into the data pointer
    *fd = open(path, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float); // position the pointer to the start of the parameter data
    memory_map_weights(weights, config, weights_ptr);
}

void build_model(Model* m, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &m->config, &m->weights, &m->fd, &m->data, &m->file_size);
    // allocate the RunState buffers
    malloc_run_state(&m->state, m->config.dim);
}

void free_model(Model* m) {
    // close the memory mapping
    if (m->data != MAP_FAILED) { munmap(m->data, m->file_size); }
    if (m->fd != -1) { close(m->fd); }
    // free the RunState buffers
    free_run_state(&m->state);
}

void linear(float* xout, float* x, float* w, float* b, int n, int d) {
    // linear layer: w(d,n) @ x (n,) + b(d,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val + b[i];
    }
}

void linear_with_relu(float* xout, float* x, float* w, float* b, int n, int d) {
    // linear layer with ReLU activation: w(d,n) @ x (n,) + b(d,) -> xout (d,)
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = fmax(val + b[i], 0.0f);
    }
}

void matmul_conv_with_relu(float* xout, float* x, float* w, float* b, float chan, int n, int d) {
    // w (c,1,n) @ x (1,n,d) + b(c,) -> xout (c,d)
    for (int c = 0; c < chan; c++) {
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[c*n + j] * x[j*d + i];
            }
            xout[c*d + i] = fmax(val + b[c], 0.0f);
        }
    }
}

float im2col_get_pixel(float* im, int height, int width, int channels, int row, int col, int channel, int pad) {
    row -= pad;
    col -= pad;
    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col_cpu(float* col, float* im, int chan, int height, int width, int ksize, int stride, int pad) {
    // im (nchan, height, width) -> col (col_size, out_height * out_width)
    int c,h,w;
    int out_height = (height + 2*pad - ksize) / stride + 1;
    int out_width = (width + 2*pad - ksize) / stride + 1;

    int col_size = chan * ksize * ksize;
    for (c = 0; c < col_size; c++) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < out_height; h++) {
            for (w = 0; w < out_width; w++) {
                int input_row = h_offset + h * stride;
                int input_col = w_offset + w * stride;
                int col_index = (c * out_height + h) * out_width + w;
                col[col_index] = im2col_get_pixel(im, height, width, chan, input_row, input_col, c_im, pad);
            }
        }
    }
}

void conv_with_reluv0(float* xout, float* x, float* w, float* b, int height, int width, int inchan, int outchan,
                        int ksize) {
    // first version of CNN layer with relu with stride = 1 and padding = ksize / 2
    // start indices are determined to reduce computational cost
    int pad = ksize / 2;
    int hw = height * width;
    for (int oc = 0; oc < outchan; oc++) {
        int xout_idx = oc * hw; // start index for xout
        int w_idx = oc * inchan * ksize * ksize; // start index fo w_idx
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                float val = 0.0f;
                for (int ic = 0; ic < inchan; ic++) {
                    int x_idx = ic * hw; // start index for x
                    int w_idx2 = ic * ksize * ksize; // additional start index for weights
                        for (int ki = 0; ki < ksize; ki++) {
                            for (int kj = 0; kj < ksize; kj++) {
                                int input_row = i - pad + ki;
                                int input_col = j - pad + kj;
                                if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                                    val += x[x_idx + input_row*width + input_col] * w[w_idx + w_idx2 + ki*ksize + kj];
                                }
                            }
                        }
                }
                xout[xout_idx + i*width + j] = fmax(0, val + b[oc]);
            }
        }
    }
}

void maxpool(float* x, int height, int width, int nchan, int ksize) {
    int out_height = height / ksize;
    int out_width = width / ksize;
    for (int c = 0; c < nchan; c++) {
        int xout_idx = c * out_height * out_width; // start index for x
        for (int i = 0; i < out_height; i++) {
            for (int j = 0; j < out_width; j++) {
                float cmax = 0;
                int x_idx = c*height*width + 2*(i*width + j); // start index for x
                for (int ki = 0; ki < ksize; ki++) {
                    for (int kj = 0; kj < ksize; kj++) {
                        cmax = fmax(cmax, x[x_idx + ki*width + kj]);
                    }
                }
                x[xout_idx + i*out_width + j] = cmax;
            }
        }
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
    Weights* w = &m->weights;
    int dim = m->config.dim;
    int nclass = m->config.nclass;
    Runstate* s = &m->state;
    float *x = s->x;
    float *x2 = s->x2;

    normalize(x2, image);
    im2col_cpu(x, x2, 1, 28, 28, 3, 1, 1);
    matmul_conv_with_relu(x2, x, w->wc1, w->bc1, 4, 3*3, 28*28);
    maxpool(x2, 28, 28, 4, 2);
    im2col_cpu(x, x2, 4, 14, 14, 3, 1, 1);
    matmul_conv_with_relu(x2, x, w->wc2, w->bc2, 8, 4*3*3, 14*14);
    maxpool(x2, 14, 14, 8, 2);
    linear_with_relu(x, x2, w->wf1, w->bf1, 8*7*7, dim);
    linear(x2, x, w->wf2, w->bf2, dim, nclass);
    softmax(x2, nclass);
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
        forward(&model, image); // output (nclass,) is stored in model.state.x2
        for (int j = 0; j < model.config.nclass; j++) {
            printf("%f\t", model.state.x2[j]);
        }
        printf("\n");
    }

    free(image);
    free_model(&model);
    return 0;
}
