
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

    normalize(x, image);
    linear_with_relu(x2, x, w->wi, w->bi, IMAGE_SZ, dim);
    linear_with_relu(x, x2, w->wh, w->bh, dim, dim/2);
    linear(x2, x, w->wo, w->bo, dim/2, nclass);
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
