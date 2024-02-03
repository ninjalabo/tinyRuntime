
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

typedef struct {
    float* wi; // input layer weights (dim,)
    float* wh; // hidden layer weights (dim, dim)
    float* wo; // output layer weights (dim,)
    float* bi; // input layer bias (dim,)
    float* bh; // hidden layer bias (dim,)
    float* bo; // output layer bias (1,)
} Weights;

typedef struct {
    float *x; // buffer (dim,)
    float *x2; // buffer (dim,)
} Runstate;

typedef struct {
    int dim;  // model dimension
    Weights weights; // the weights of the model
    Runstate state; // the current state in the forward pass
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    size_t file_size; // size of the checkpoint file in bytes
} Model;

void malloc_run_state(Runstate* s, int dim) {
    s->x = calloc(dim, sizeof(float));
    s->x2 = calloc(dim, sizeof(float));
}

void free_run_state(Runstate* s) {
    free(s->x);
    free(s->x2);
}

void memory_map_weights(Weights* w, int dim, float* ptr) {
    // maps memory for the weights and bias of each layer
    w->wi = ptr;
    ptr += dim;
    w->wh = ptr;
    ptr += dim * dim;
    w->wo = ptr;
    ptr += dim;
    w->bi = ptr;
    ptr += dim;
    w->bh = ptr;
    ptr += dim;
    w->bo = ptr;
}

void read_checkpoint(char* path, int* dim, Weights* weigths, int* fd, float** data, size_t* file_size) {
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", path); exit(EXIT_FAILURE); }
    // read model config
    if (fread(dim, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); } 
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the model weights into the data pointer
    *fd = open(path, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + 256/sizeof(float);
    memory_map_weights(weigths, *dim, weights_ptr);
}
void build_model(Model* m, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &m->dim, &m->weights, &m->fd, &m->data, &m->file_size);
    // allocate the RunState buffers
    malloc_run_state(&m->state, m->dim);
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

float forward_one(Model* m, float input) {
    // forward step for one input
    Weights* w = &m->weights;
    Runstate* s = &m->state;
    float *x = s->x;
    float *x2 = s->x2;
    int dim = m->dim;
    float output;

    matmul_relu(x, &input, w->wi, w->bi, 1, dim);
    matmul_relu(x2, x, w->wh, w->bh, dim, dim);
    matmul(&output, x2, w->wo, w->bo, dim, 1);

    return output;
}

float* forward_all(Model* m, float* inputs, int size) {
    // forward step for all inputs
    float* outputs = calloc(size, sizeof(float));
    for (int i = 0; i<size; i++) {
        outputs[i] = forward_one(m, inputs[i]);
    }
    return outputs;
}

void write_weights(Model* m, char* filepath) {
    // writes a file to check if the model parameters are read correctly
    int dim = m->dim;
    FILE* f = fopen(filepath, "w");
    for (int i = 0; i<dim; i++) { fprintf(f, "%f\n", m->weights.wi[i]); }
    for (int i = 0; i<dim*dim; i++) { fprintf(f, "%f\n", m->weights.wh[i]); }
    for (int i = 0; i<dim; i++) { fprintf(f, "%f\n", m->weights.wo[i]); }
    for (int i = 0; i<dim; i++) { fprintf(f, "%f\n", m->weights.bi[i]); }
    for (int i = 0; i<dim; i++) { fprintf(f, "%f\n", m->weights.bh[i]); }
    fprintf(f, "%f\n", m->weights.bo[0]);
    fclose(f); 
}

void write_output(float* outputs, char* filepath, int size) {
    // writes a file to check if the model parameters are read correctly
    FILE* f = fopen(filepath, "w");
    for (int i = 0; i<size; i++)
        fprintf(f, "%f\n", outputs[i]);
    fclose(f); 
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

    // build the model via the model .bin file 
    Model model;
    build_model(&model, checkpoint_path);

    // run
    float inputs[4] = {0.001, 1.57, 3.14, 6.28};
    float* outputs = forward_all(&model, inputs, 4);

    // check the weights and outputs are valid
    //write_weights(&model, "weightsc.txt");
    //write_output(outputs, "outputc.txt", 4);
    
    for (int i = 0; i<4; i++)
        printf("%f\t", outputs[i]);

    // memory and file handles cleanup
    free(outputs);
    free_model(&model);
    return 0;
}
