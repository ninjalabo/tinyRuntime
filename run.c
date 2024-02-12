
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

uint8_t matmul_final(float* x, float* w, float* b, int n, int d) {
    // argmax(W (d,n) @ x (n,) + b(d,))
    // calculate matmul and return prediction from the resulting one hot encoded vector
    float cmax = 0;
    uint8_t index = 0;
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        val += b[i];
        if (val > cmax) {
            cmax = val;
            index = (uint8_t) i;
        }
    }
    return index;
}

void write_activation(float* x, int size) {
    FILE* f = fopen("tmp.txt", "a");
    for (int i = 0; i < size; i++)
        fprintf(f, "%f\n", x[i]);
    fclose(f);
}

void normalize(float* xout, uint8_t* input) {
    for (int i = 0; i < 28*28; i++) {
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
    float *x2 = s->x2;
    uint8_t out;

    normalize(x, input);
    matmul_relu(x2, x, w->wi, w->bi, IMAGE_SZ, dim);
    matmul_relu(x, x2, w->wh, w->bh, dim, dim/2);
    out = matmul_final(x, w->wo, w->bo, dim/2, nclass);

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
    if (rows != cols || rows != 28) { fprintf(stderr, "Number of rows and cols doesn't match\n"); exit(EXIT_FAILURE); }
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
    for (int i = 0; i < IMAGE_SZ*dim; i++) { fprintf(f, "%.12f\n", m->weights.wi[i]); }
    for (int i = 0; i < dim*dim/2; i++) { fprintf(f, "%.12f\n", m->weights.wh[i]); }
    for (int i = 0; i < dim*nclass/2; i++) { fprintf(f, "%.12f\n", m->weights.wo[i]); }
    for (int i = 0; i < dim; i++) { fprintf(f, "%.12f\n", m->weights.bi[i]); }
    for (int i = 0; i < dim/2; i++) { fprintf(f, "%.12f\n", m->weights.bh[i]); }
    for (int i = 0; i < nclass; i++) { fprintf(f, "%.12f\n", m->weights.bo[i]); }
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
    //write_params(&model, "weightsc.txt");
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
