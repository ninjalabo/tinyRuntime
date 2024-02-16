
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
    float* wi; // input layer weights (dim, IMAGE_SZ)
    float* wh; // hidden layer weights (dim/2, dim)
    float* wo; // output layer weights (nclass, dim/2)
    float* bi; // input layer bias (dim,)
    float* bh; // hidden layer bias (dim/2,)
    float* bo; // output layer bias (nclass,)
} Weights;

typedef struct {
    float* x; // buffer to store the output of a layer (IMAGE_SZ,)
    float* x2; // buffer to store the output of a layer (dim,)
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
    s->x = calloc(IMAGE_SZ, sizeof(float));
    s->x2 = calloc(dim, sizeof(float));
}

void free_run_state(Runstate* s) {
    free(s->x);
    free(s->x2);
}

void memory_map_weights(Weights* w, Config* c, float* ptr) {
    // maps memory for the weights and bias of each layer
    int dim = c->dim;
    int nclass = c->nclass;
    w->wi = ptr;
    ptr += IMAGE_SZ * dim;
    w->wh = ptr;
    ptr += dim * dim/2;
    w->wo = ptr;
    ptr +=  nclass * dim/2;
    w->bi = ptr;
    ptr += dim;
    w->bh = ptr;
    ptr += dim/2;
    w->bo = ptr;
}

void read_checkpoint(char* path, Config* config, Weights* weigths, int* fd, float** data, size_t* file_size) {
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
    memory_map_weights(weigths, config, weights_ptr);
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

void matmul(float* xout, float* x, float* w, float* b, int n, int d) {
    // W (d,n) @ x (n,) + b(d,) -> xout (d,)
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

void matmul_relu(float* xout, float* x, float* w, float* b, int n, int d) {
    // W (d,n) @ x (n,) + b(d,) -> xout (d,)
    // matmul with ReLU
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

void normalize(float* xout, uint8_t* input) {
    for (int i = 0; i < 28*28; i++) {
        xout[i] = ((float) input[i] / 255 - 0.5) / 0.5;
    }
}

// sanity check function
void write_activation(float* x, int size) {
    FILE* f = fopen("actc.txt", "w");
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
    float *x2 = s->x2;

    normalize(x, input);
    matmul_relu(x2, x, w->wi, w->bi, IMAGE_SZ, dim);
    matmul_relu(x, x2, w->wh, w->bh, dim, dim/2);
    matmul(x2, x, w->wo, w->bo, dim/2, nclass);

    return x2;
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
                    forward_one(m, image);  // now model output (nclass,) is stored in model.state.x2
                    outputs[i] = find_max(m->state.x2, m->config.nclass);
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
    forward_one(m, image);  // now model output (nclass,) is stored in model.state.x2
    softmax(m->state.x2, m->config.nclass);
}

// sanity check function
void write_params(Model* m, char* filepath) {
    int dim = m->config.dim;
    int nclass = m->config.nclass;
    FILE* f = fopen(filepath, "w");
    fwrite(m->weights.wi, sizeof(float), IMAGE_SZ*dim, f);
    fwrite(m->weights.wh, sizeof(float), dim*dim/2, f);
    fwrite(m->weights.wo, sizeof(float), dim*nclass/2, f);
    fwrite(m->weights.bi, sizeof(float), dim, f);
    fwrite(m->weights.bh, sizeof(float), dim/2, f);
    fwrite(m->weights.bo, sizeof(float), nclass, f);
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
            printf("%.12f\n", model.state.x2[i]);
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
