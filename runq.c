
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

void normalize(float* xout, uint8_t* input) {
    for (int i = 0; i < IMAGE_SZ; i++) {
        xout[i] = ((float) input[i] / 255 - 0.5) / 0.5;
    }
}

// sanity check function
void write_activation(float* x, int size) {
    FILE* f = fopen("actcq8.txt", "w");
    for (int i = 0; i < size; i++)
        fprintf(f, "%f\n", x[i]);
    fclose(f);
}

float* forward_one(Model* m, uint8_t* input) {
    // forward step for one input
    Weights* w = &m->weights;
    int dim = m->config.dim;
    int nclass = m->config.nclass;
    Runstate* s = &m->state;
    float *x = s->x;
    QuantizedTensor xq = s->xq;

    normalize(x, input);
    quantize(&xq, x, IMAGE_SZ);
    matmul_relu(x, &xq, w->wi, w->bi, IMAGE_SZ, dim);
    quantize(&xq, x, dim);
    matmul_relu(x, &xq, w->wh, w->bh, dim, dim/2);
    quantize(&xq, x, dim/2);
    matmul(x, &xq, w->wo, w->bo, dim/2, nclass);

    return x;
}

uint8_t find_max(float* x, int nclass) {
    // find the maximum value of one hot encoded vector x (nclass,)
    float cmax = 0;
    uint8_t index = 0;
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < nclass; i++) {
        if (x[i] > cmax) {
            cmax = x[i];
            index = (uint8_t) i;
        }
    }
    return index;
}

void read_mnist_image(char* path, uint8_t *image) {
    FILE* file = fopen(path, "rb");
    if (!file) { perror("Error opening file"); exit(EXIT_FAILURE); }
    // allocate memory for the images array and read images
    fread(image, sizeof(uint8_t), IMAGE_SZ, file);
    fclose(file);
}

void classify(Model* m, const char* dir_path, uint8_t* outputs, uint8_t* image, int* nimages) {
    DIR* dir;
    struct dirent* entry;
    if ((dir = opendir(dir_path)) != NULL) {
        int i = 0;
        while (i < *nimages) {
            if ((entry = readdir(dir)) != NULL) {
                if (entry->d_type == DT_REG) {
                    char file_path[100];
                    snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, entry->d_name);
                    read_mnist_image(file_path, image);
                    forward_one(m, image);  // now model output (nclass,) is stored in model.state.x
                    outputs[i] = find_max(m->state.x, m->config.nclass);
                    i += 1;
                }
            } else { *nimages = i; }
        }
        closedir(dir);
    } else { printf("%s", dir_path);
        perror("Unable to open directory"); exit(EXIT_FAILURE); }
}

float evaluate_model(Model* m, const char* base_path, uint8_t* image) {
    float acc = 0;
    int tot = 0;
    int nimages = 1200; // 1200 because all directories contains <1200 files
    uint8_t* outputs = malloc(nimages);
    for (int i = 0; i < 10; i++) {
        char dir_path[100];
        snprintf(dir_path, sizeof(dir_path), "%s/%d", base_path, i);
        nimages = 1200; // initialize at every step
        classify(m, dir_path, outputs, image, &nimages);
        tot += nimages;
        for (int j = 0; j < nimages; j++) {
            if (outputs[j] == i)
                acc += 1;
        }
    }
    free(outputs);
    acc /= tot;
    return acc;
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

void test_model(Model* m, uint8_t* image) {
    char* file_path = "data/MNIST/sorted/7/1012";
    read_mnist_image(file_path, image);
    forward_one(m, image);  // now model output (nclass,) is stored in model.state.x
    softmax(m->state.x, m->config.nclass);
}

// sanity check function
void write_params(Model* m, char* filepath) {
    int dim = m->config.dim;
    int nclass = m->config.nclass;
    FILE* f = fopen(filepath, "wb");
    // write model parameters and scaling factors
    fwrite(&(m->config), sizeof(Config), 1, f);
    fwrite(&GS, sizeof(int), 1, f);
    fwrite(m->weights.wi->q, 1, IMAGE_SZ * dim, f);
    fwrite(m->weights.wi->s, sizeof(float), IMAGE_SZ*dim/GS, f);
    fwrite(m->weights.wh->q, 1, dim*dim/2, f);
    fwrite(m->weights.wh->s, sizeof(float), dim*dim/2/GS, f);
    fwrite(m->weights.wo->q, 1, dim*nclass/2, f);
    fwrite(m->weights.wo->s, sizeof(float), dim*nclass/2/GS, f);
    fwrite(m->weights.bi->q, 1, dim, f);
    fwrite(m->weights.bi->s, sizeof(float), dim/GS, f);
    fwrite(m->weights.bh->q, 1, dim/2, f);
    fwrite(m->weights.bh->s, sizeof(float), dim/2/GS, f);
    fwrite(m->weights.bo->q, 1, nclass, f);
    fwrite(m->weights.bo->s, sizeof(float), 1, f);
    fclose(f);
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint>\n");
    fprintf(stderr, "Example: run model.bin -c 5 -n 10\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -c <int>       class in {0,1,2,...,9}, default 0\n");
    fprintf(stderr, "  -n <int>       number of images to classify, default 5\n");
    fprintf(stderr, "  -m <string>    mode: classify|test, default classify\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    int nimages = 5;  // number of images to classify
    uint8_t label = 0;  // image class to classify
    char* mode = "classify";

    // read a model path, number of images and label to classify if given
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 'c') {
            int val = atoi(argv[i + 1]);
            if (val >= 0 && val <= 9) { label = val; } else { error_usage(); }
        }
        else if (argv[i][1] == 'n') { nimages = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else { error_usage(); }
    }

    Model model;
    build_model(&model, checkpoint_path);

    char* base_path = "data/MNIST/sorted";
    uint8_t* image = malloc(IMAGE_SZ);

    if (strcmp(mode, "classify") == 0) {
        char dir_path[100];
        snprintf(dir_path, sizeof(dir_path), "%s/%d", base_path, label);
        uint8_t* outputs = malloc(nimages);
        classify(&model, dir_path, outputs, image, &nimages);
        for (int i = 0; i < nimages; i++) {
            printf("%i\n", outputs[i]);
        }
        free(outputs);
    } else if (strcmp(mode, "classifyall") == 0) {
        float acc = evaluate_model(&model, base_path, image);
        printf("Accuracy: %f\n", acc);
    } else if (strcmp(mode, "test") == 0) {
        test_model(&model, image);
        for (int i = 0; i < model.config.nclass; i++) {
            printf("%.12f\n", model.state.x[i]);
        }
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free(image);
    free_model(&model);
    return 0;
}
