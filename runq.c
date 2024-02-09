
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

int GS = 0; // group size global for quantization of the weights

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

typedef struct {
    QuantizedTensor* wi; // input layer weights (dim,)
    QuantizedTensor* wh; // hidden layer weights (dim, dim)
    QuantizedTensor* wo; // output layer weights (dim,)
    QuantizedTensor* bi; // input layer bias (dim,)
    QuantizedTensor* bh; // hidden layer bias (dim,)
    QuantizedTensor* bo; // output layer bias (1,)
} Weights;

typedef struct {
    float *x; // buffer (dim,)
    QuantizedTensor xq; // quantized x (dim,)
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
    // FIX .s calloc(dim/GS, sizeof(float))
    s->xq = (QuantizedTensor) { .q = calloc(dim, sizeof(int8_t)), .s = calloc(dim, sizeof(float)) };}

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
    for(int i=0; i<n; i++) {
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

void memory_map_weights(Weights* w, int dim, void* ptr) {
    // maps memory for the weights and bias of each layer
    w->wi = init_quantized_tensors(&ptr, 1, dim);
    w->wh = init_quantized_tensors(&ptr, 1, dim*dim);;
    w->wo = init_quantized_tensors(&ptr, 1, dim);
    w->bi = init_quantized_tensors(&ptr, 1, dim);
    w->bh = init_quantized_tensors(&ptr, 1, dim);
    w->bo = init_quantized_tensors(&ptr, 1, 1);
}

void read_checkpoint(char* path, int* dim, Weights* weigths, int* fd, float** data, size_t* file_size) {
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", path); exit(EXIT_FAILURE); }
    // read model config
    if (fread(dim, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); } 
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
    void* weights_ptr = *data + 256/sizeof(float);
    memory_map_weights(weigths, *dim, weights_ptr);
}
void build_model(Model* m, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &m->dim, &m->weights, &m->fd, &m->data, &m->file_size);
    // allocate the RunState buffers
    malloc_run_state(&m->state, m->dim);
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

float forward_one(Model* m, float input) {
    // forward step for one input
    Weights* w = &m->weights;
    Runstate* s = &m->state;
    int dim = m->dim;
    float output;
    
    // calculate activation of the first layer (1,) @ (GS,) + (GS,) -> (GS,)
    for (int i = 0; i<dim; i++)
        s->x[i] = fmax(((float) w->wi->q[i]) * w->wi->s[i / GS] * input + ((float) w->bi->q[i]) * w->bi->s[i / GS], 0.0f);
    // quantize float type activations and matmul them with model parameters  
    quantize(&s->xq, s->x, dim);
    matmul_relu(s->x, &s->xq, w->wh, w->bh, dim, dim);
    quantize(&s->xq, s->x, dim);
    matmul(&output, &s->xq, w->wo, w->bo, dim, 1);

    return output;
}

float* forward_all(Model* m, float* inputs, int size) {
    // forward step for all inputs
    float* outputs = malloc(size*sizeof(float));
    for (int i = 0; i<size; i++) {
        outputs[i] = forward_one(m, inputs[i]);
    }
    return outputs;
}

void write_params(Model* m, char* filepath) {
    // writes a file to check if the model parameters are read correctly
    int dim = m->dim;
    FILE* f = fopen(filepath, "w");
    for (int i = 0; i<dim; i++) { fprintf(f, "%i\n", m->weights.wi->q[i]); }
    for (int i = 0; i<dim*dim; i++) { fprintf(f, "%i\n", m->weights.wh->q[i]); }
    for (int i = 0; i<dim; i++) { fprintf(f, "%i\n", m->weights.wo->q[i]); }
    for (int i = 0; i<dim; i++) { fprintf(f, "%i\n", m->weights.bi->q[i]); }
    for (int i = 0; i<dim; i++) { fprintf(f, "%i\n", m->weights.bh->q[i]); }
    fprintf(f, "%i\n\n", m->weights.bo->q[0]);
    fprintf(f, "%f\n", m->weights.wi->s[0]);
    for (int i = 0; i<dim*dim/GS; i++) { fprintf(f, "%f\n", m->weights.wh->s[i]); }
    fprintf(f, "%f\n", m->weights.wo->s[0]);
    fprintf(f, "%f\n", m->weights.bi->s[0]);
    fprintf(f, "%f\n", m->weights.bh->s[0]);
    fprintf(f, "%f\n", m->weights.bo->s[0]);
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

    // build the model via model.bin file 
    Model model;
    build_model(&model, checkpoint_path);

    // run
    float inputs[4] = {0.001, 1.57, 3.14, 6.28};
    float* outputs = forward_all(&model, inputs, 4);

    // check the weights and outputs are valid
    //write_params(&model, "paramsq8.txt");
    //write_output(outputs, "outputq8.txt", 4);
    
    for (int i = 0; i<4; i++)
        printf("%f\t", outputs[i]);

    // memory and file handles cleanup
    free(outputs);
    free_model(&model);
    return 0;
}
