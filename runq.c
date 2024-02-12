
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

#define IMAGE_SZ (28*28)

int GS = 0; // group size global for quantization of the weights

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

typedef struct {
    int dim; // model dimension
    int nclass; // the number of classes
} Config;

typedef struct {
    QuantizedTensor* wi; // input layer weights (dim, IMAGE_SZ)
    QuantizedTensor* wh; // hidden layer weights (dim/2, dim)
    QuantizedTensor* wo; // output layer weights (nclass, dim/2)
    QuantizedTensor* bi; // input layer bias (dim,)
    QuantizedTensor* bh; // hidden layer bias (dim/2,)
    QuantizedTensor* bo; // output layer bias (nclass,)
} Weights;

typedef struct {
    float* x; // buffer to store the output of a layer (IMAGE_SZ,)
    QuantizedTensor xq; // buffer for quantized arrays
} Runstate;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    Weights weights; // the weights of the model
    Runstate state; // the current state in the forward pass
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    size_t file_size; // size of the checkpoint file in bytes
} Model;

void malloc_run_state(Runstate* s) {
    int inp_dim = IMAGE_SZ;
    s->x = calloc(inp_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(inp_dim, sizeof(int8_t)), .s = calloc(inp_dim/GS + 1, sizeof(float)) };
}

void free_run_state(Runstate* s) {
    free(s->x);
    free(s->xq.q);
    free(s->xq.s);
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {
        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }
        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i < n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void memory_map_weights(Weights* w, Config* c, void* ptr) {
    // maps memory for the weights and bias of each layer
    int dim = c->dim;
    int nclass = c->nclass;
    w->wi = init_quantized_tensors(&ptr, 1, IMAGE_SZ*dim);
    w->wh = init_quantized_tensors(&ptr, 1, dim/2*dim);;
    w->wo = init_quantized_tensors(&ptr, 1, dim/2*nclass);
    w->bi = init_quantized_tensors(&ptr, 1, dim);
    w->bh = init_quantized_tensors(&ptr, 1, dim/2);
    w->bo = init_quantized_tensors(&ptr, 1, nclass);
}

void read_checkpoint(char* path, Config* config, Weights* weigths, int* fd, float** data, size_t* file_size) {
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", path); exit(EXIT_FAILURE); }
    // read model config
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    int group_size; // the group size used in quantization
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    GS = group_size; // set as global, as it will be used in many places
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the model weights into the data pointer
    *fd = open(path, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    void* weights_ptr = *data + (sizeof(Config)+sizeof(int))/sizeof(float); // position the pointer to the start of the parameter data
    memory_map_weights(weigths, config, weights_ptr);
}

void build_model(Model* m, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &m->config, &m->weights, &m->fd, &m->data, &m->file_size);
    // allocate the RunState buffers
    malloc_run_state(&m->state);
}

void free_model(Model* m) {
    // free QuantizedTensors
    free(m->weights.wi);
    free(m->weights.wh);
    free(m->weights.wo);
    free(m->weights.bi);
    free(m->weights.bh);
    free(m->weights.bo);
    // close the memory mapping
    if (m->data != MAP_FAILED) { munmap(m->data, m->file_size); }
    if (m->fd != -1) { close(m->fd); }
    // free the RunState buffers
    free_run_state(&m->state);
}

void matmul(float* xout, QuantizedTensor* x, QuantizedTensor* w, QuantizedTensor* b, int n, int d) {
    // W (d,n) @ x (n,) + b(d,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        // do the matmul in groups of GS
        for (int j = 0; j < n; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        xout[i] = val + ((float) b->q[i]) * b->s[i / GS];
    }
}

void matmul_relu(float* xout, QuantizedTensor* x, QuantizedTensor* w, QuantizedTensor* b, int n, int d) {
    // W (d,n) @ x (n,) + b(d,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        // do the matmul in groups of GS
        for (int j = 0; j < n; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        xout[i] = fmax(0.0f, val + ((float) b->q[i]) * b->s[i / GS]);
    }
}

uint8_t matmul_final(QuantizedTensor* x, QuantizedTensor* w, QuantizedTensor* b, int n, int d) {
    // argmax(W (d,n) @ x (n,) + b(d,))
    // calculate matmul and return prediction from the resulting one hot encoded vector
    float cmax = 0;
    uint8_t index = 0;
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        // do the matmul in groups of GS
        for (int j = 0; j < n; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        val += ((float) b->q[i]) * b->s[i / d];
        if (val > cmax) {
            cmax = val;
            index = (uint8_t) i;
        }
    }
    return index;
}

void normalize(float* xout, uint8_t* input) {
    for (int i = 0; i < IMAGE_SZ; i++) {
        xout[i] = ((float) input[i] / 255 - 0.5) / 0.5;
    }
}

uint8_t forward_one(Model* m, uint8_t* input) {
    // forward step for one input
    Weights* w = &m->weights;
    int dim = m->config.dim;
    int nclass = m->config.nclass;
    Runstate* s = &m->state;
    float *x = s->x;
    QuantizedTensor xq = s->xq;
    uint8_t out;

    normalize(x, input);
    quantize(&xq, x, IMAGE_SZ);
    matmul_relu(x, &xq, w->wi, w->bi, IMAGE_SZ, dim);
    quantize(&xq, x, dim);
    matmul_relu(x, &xq, w->wh, w->bh, dim, dim/2);
    quantize(&xq, x, dim/2);
    out = matmul_final(&xq, w->wo, w->bo, dim/2, nclass);

    return out;
}

uint8_t* forward_all(Model* m, uint8_t** inputs, int size) {
    // forward step for all inputs
    uint8_t* outputs = malloc(size);
    for (int i = 0; i < size; i++) {
        outputs[i] = forward_one(m, inputs[i]);
    }
    return outputs;
}

void read_mnist_images(char* path, uint8_t ***images, int *nimages) {
    FILE* file = fopen(path, "rb");
    if (!file) { perror("Error opening file"); exit(EXIT_FAILURE); }
    // check if the magic number corresponds to MNIST image files
    unsigned int magic_number;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    magic_number = ntohl(magic_number);
    if (magic_number != 0x00000803) { fprintf(stderr, "Invalid magic number\n"); exit(EXIT_FAILURE); }
    // read number of images
    fread(nimages, sizeof(uint32_t), 1, file);
    *nimages = ntohl(*nimages);
    // read number of rows and columns in each image
    int rows; int cols;
    fread(&rows, sizeof(uint32_t), 1, file);
    fread(&cols, sizeof(uint32_t), 1, file);
    rows = ntohl(rows);
    cols = ntohl(cols);
    // allocate memory for the images array and read images
    *images = malloc(sizeof(uint8_t*) * (*nimages));
    for (int i = 0; i < *nimages; i++) {
        (*images)[i] = malloc(sizeof(uint8_t) * rows * cols);
        fread((*images)[i], sizeof(uint8_t), rows * cols, file);
    }
    fclose(file);
}

void read_mnist_labels(char* path, uint8_t **labels, int *nlabels) {
    FILE* file = fopen(path, "rb");
    if (!file) { perror("Error opening file"); exit(EXIT_FAILURE); }
    // check if the magic number corresponds to MNIST label files
    unsigned int magic_number;
    fread(&magic_number, sizeof(uint32_t), 1, file);
    magic_number = ntohl(magic_number);
    if (magic_number != 0x00000801) { fprintf(stderr, "%d", magic_number); exit(EXIT_FAILURE); }
    // read the number of labels
    fread(nlabels, sizeof(uint32_t), 1, file);
    *nlabels = ntohl(*nlabels);
    // allocate memory for the labels array and read labels
    *labels = malloc(sizeof(uint8_t) * (*nlabels));
    fread(*labels, sizeof(uint8_t), *nlabels, file);
    fclose(file);
}

void write_params(Model* m, char* filepath) {
    // writes a file to check if the model parameters are read correctly
    int dim = m->config.dim;
    int nclass = m->config.nclass;
    FILE* f = fopen(filepath, "w");
    // write model parameters
    for (int i = 0; i < IMAGE_SZ*dim; i++) { fprintf(f, "%i\n", m->weights.wi->q[i]); }
    for (int i = 0; i < dim*dim/2; i++) { fprintf(f, "%i\n", m->weights.wh->q[i]); }
    for (int i = 0; i < dim*nclass/2; i++) { fprintf(f, "%i\n", m->weights.wo->q[i]); }
    for (int i = 0; i < dim; i++) { fprintf(f, "%i\n", m->weights.bi->q[i]); }
    for (int i = 0; i < dim/2; i++) { fprintf(f, "%i\n", m->weights.bh->q[i]); }
    for (int i = 0; i < nclass; i++) { fprintf(f, "%i\n", m->weights.bo->q[i]); }
    // write scale factors
    for (int i = 0; i < IMAGE_SZ*dim/GS; i++) { fprintf(f, "%.12f\n", m->weights.wi->s[i]); }
    for (int i = 0; i < dim*dim/2/GS; i++) { fprintf(f, "%.12f\n", m->weights.wh->s[i]); }
    for (int i = 0; i < dim*nclass/2/GS; i++) { fprintf(f, "%.12f\n", m->weights.wo->s[i]); }
    for (int i = 0; i < dim/GS; i++) { fprintf(f, "%.12f\n", m->weights.bi->s[i]); }
    for (int i = 0; i < dim/2/GS; i++) { fprintf(f, "%.12f\n", m->weights.bh->s[i]); }
    fprintf(f, "%.12f\n", m->weights.bo->s[0]);
    fclose(f); 

}

void write_output(uint8_t* outputs, char* filepath, int size) {
    // writes a file to check if the model parameters are read correctly
    FILE* f = fopen(filepath, "w");
    for (int i = 0; i < size; i++)
        fprintf(f, "%i\n", outputs[i]);
    fclose(f); 
}

void print_accuracy(uint8_t* outputs, uint8_t* labels, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (outputs[i]==labels[i])
            count += 1;
    }
    printf("Accuracy: %.2f %%", 100 * (float) count / n);
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint>\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin

    // read a model path
    if (argc == 2) { checkpoint_path = argv[1]; } else { error_usage(); }

    // read images
    char* input_path = "data/MNIST/raw/t10k-images-idx3-ubyte";
    int nimages;
    uint8_t** images;
    read_mnist_images(input_path, &images, &nimages);
    // read labels
    input_path = "data/MNIST/raw/t10k-labels-idx1-ubyte";
    int nlabels;
    uint8_t* labels;
    read_mnist_labels(input_path, &labels, &nlabels);

    if (nlabels!=nimages) {
        fprintf(stderr, "Number of labels (%d) does not match the number of images (%d)\n", nlabels, nimages);
        exit(EXIT_FAILURE);
    }

    // build the model via model.bin file 
    Model model;
    build_model(&model, checkpoint_path);

    // run
    uint8_t* outputs = forward_all(&model, images, nimages);

    // check the weights and outputs are valid
    //write_weights(&model, "weightsc.txt");
    //write_output(outputs, "outputc.txt", nimages);
    print_accuracy(outputs, labels, nlabels);

    // memory and file handles cleanup
    for (int i = 0; i < nimages; i++) {
        free(images[i]);
    }
    free(images);
    free(labels);
    free(outputs);
    free_model(&model);
    return 0;
}
