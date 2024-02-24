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

static int GS;			// group size global for quantization of the weights

typedef struct {
	int8_t *q;		// quantized values
	float *s;		// scaling factors
} QuantizedTensor;

typedef struct {
	int dim;		// model dimension
	int nclass;		// the number of classes
} Config;

typedef struct {
	QuantizedTensor *wi;	// input layer weights (dim, IMAGE_SZ)
	QuantizedTensor *wh;	// hidden layer weights (dim/2, dim)
	QuantizedTensor *wo;	// output layer weights (nclass, dim/2)
	QuantizedTensor *bi;	// input layer bias (dim,)
	QuantizedTensor *bh;	// hidden layer bias (dim/2,)
	QuantizedTensor *bo;	// output layer bias (nclass,)
} Weights;

typedef struct {
	float *x;		// buffer to store the output of a layer (IMAGE_SZ,)
	QuantizedTensor xq;	// buffer for quantized arrays
} Runstate;

typedef struct {
	Config config;		// the hyperparameters of the architecture (the blueprint)
	Weights weights;	// the weights of the model
	Runstate state;		// the current state in the forward pass
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

static void malloc_run_state(Runstate *s)
{
	int inp_dim = IMAGE_SZ;
	s->x = calloc(inp_dim, sizeof(float));
	s->xq = (QuantizedTensor) {
	.q = calloc(inp_dim, sizeof(int8_t)),.s =
		    calloc(inp_dim / GS + 1, sizeof(float))};
}

static void free_run_state(Runstate *s)
{
	free(s->x);
	free(s->xq.q);
	free(s->xq.s);
}

static void quantize(QuantizedTensor *qx, float *x, int n)
{
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
			float quant_value = x[group * GS + i] / scale;	// scale
			int8_t quantized = (int8_t) round(quant_value);	// round and clamp
			qx->q[group * GS + i] = quantized;
		}
	}
}

static QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each)
{
	void *p = *ptr;
	QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
	for (int i = 0; i < n; i++) {
		/* map quantized int8 values */
		res[i].q = (int8_t *) p;
		p = (int8_t *) p + size_each;
		/* map scale factors */
		res[i].s = (float *)p;
		p = (float *)p + size_each / GS;
	}
	*ptr = p;		// advance ptr to current position
	return res;
}

static void memory_map_weights(Weights *w, Config *c, void *ptr)
{
	// maps memory for the weights and bias of each layer
	int dim = c->dim;
	int nclass = c->nclass;
	w->wi = init_quantized_tensors(&ptr, 1, IMAGE_SZ * dim);
	w->wh = init_quantized_tensors(&ptr, 1, dim / 2 * dim);;
	w->wo = init_quantized_tensors(&ptr, 1, dim / 2 * nclass);
	w->bi = init_quantized_tensors(&ptr, 1, dim);
	w->bh = init_quantized_tensors(&ptr, 1, dim / 2);
	w->bo = init_quantized_tensors(&ptr, 1, nclass);
}

static void read_checkpoint(char *path, Config *config, Weights *weigths, int *fd,
		     float **data, size_t *file_size)
{
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "Couldn't open file %s\n", path);
		exit(EXIT_FAILURE);
	}
	// read model config
	if (fread(config, sizeof(Config), 1, file) != 1) {
		exit(EXIT_FAILURE);
	}
	int group_size;		// the group size used in quantization
	if (fread(&group_size, sizeof(int), 1, file) != 1) {
		exit(EXIT_FAILURE);
	}
	GS = group_size;	// set as global, as it will be used in many places
	// figure out the file size
	fseek(file, 0, SEEK_END);	// move file pointer to end of file
	*file_size = ftell(file);	// get the file size, in bytes
	fclose(file);
	// memory map the model weights into the data pointer
	*fd = open(path, O_RDONLY);	// open in read only mode
	if (*fd == -1) {
		fprintf(stderr, "open failed!\n");
		exit(EXIT_FAILURE);
	}
	*data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
	if (*data == MAP_FAILED) {
		fprintf(stderr, "mmap failed!\n");
		exit(EXIT_FAILURE);
	}
	void *weights_ptr = *data + (sizeof(Config) + sizeof(int)) / sizeof(float);	// position the pointer to the start of the parameter data
	memory_map_weights(weigths, config, weights_ptr);
}

static void build_model(Model * m, char *checkpoint_path)
{
	// read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, &m->config, &m->weights, &m->fd,
			&m->data, &m->file_size);
	// allocate the RunState buffers
	malloc_run_state(&m->state);
}

static void free_model(Model *m)
{
	// free QuantizedTensors
	free(m->weights.wi);
	free(m->weights.wh);
	free(m->weights.wo);
	free(m->weights.bi);
	free(m->weights.bh);
	free(m->weights.bo);
	// close the memory mapping
	if (m->data != MAP_FAILED) {
		munmap(m->data, m->file_size);
	}
	if (m->fd != -1) {
		close(m->fd);
	}
	// free the RunState buffers
	free_run_state(&m->state);
}

static void linear(float *xout, QuantizedTensor *x, QuantizedTensor *w,
	    QuantizedTensor *b, int n, int d)
{
	// linear layer: w(d,n) @ x (n,) + b(d,) -> xout (d,)
	// by far the most amount of time is spent inside this little function
	int i;
	//#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) {
		float val = 0.0f;
		int32_t ival = 0;
		int in = i * n;
		// do the matmul in groups of GS
		for (int j = 0; j < n; j += GS) {
			for (int k = 0; k < GS; k++) {
				ival +=
				    ((int32_t) x->q[j + k]) *
				    ((int32_t) w->q[in + j + k]);
			}
			val +=
			    ((float)ival) * w->s[(in + j) / GS] * x->s[j / GS];
			ival = 0;
		}
		xout[i] = val + ((float)b->q[i]) * b->s[i / GS];
	}
}

static void linear_with_relu(float *xout, QuantizedTensor *x, QuantizedTensor *w,
		      QuantizedTensor *b, int n, int d)
{
	// linear layer with ReLU activation: w(d,n) @ x (n,) + b(d,) -> xout (d,)
	// by far the most amount of time is spent inside this little function
	int i;
	//#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) {
		float val = 0.0f;
		int32_t ival = 0;
		int in = i * n;
		// do the matmul in groups of GS
		for (int j = 0; j < n; j += GS) {
			for (int k = 0; k < GS; k++) {
				ival +=
				    ((int32_t) x->q[j + k]) *
				    ((int32_t) w->q[in + j + k]);
			}
			val +=
			    ((float)ival) * w->s[(in + j) / GS] * x->s[j / GS];
			ival = 0;
		}
		xout[i] = fmax(0.0f, val + ((float)b->q[i]) * b->s[i / GS]);
	}
}

static void normalize(float *xout, uint8_t * image)
{
	// normalize values [0, 255] -> [-1, 1]
	for (int i = 0; i < IMAGE_SZ; i++) {
		xout[i] = ((float)image[i] / 255 - 0.5) / 0.5;
	}
}

static void softmax(float *x, int size)
{
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

static void read_mnist_image(char *path, uint8_t * image)
{
	size_t bytes;
	FILE *file = fopen(path, "rb");
	if (!file) {
		perror("Error opening file");
		exit(EXIT_FAILURE);
	}
	bytes = fread(image, sizeof(uint8_t), IMAGE_SZ, file);
	if (!bytes) {
		perror("Error reading file");
		exit(EXIT_FAILURE);
	}
	fclose(file);
}

#ifdef DEBUG
// sanity check function for writing tensors,
// e.g., it can be used to evaluate values after a specific layer.
static void write_tensor(float *x, int size)
{
	FILE *f = fopen("actc.txt", "w");
	for (int i = 0; i < size; i++)
		fprintf(f, "%f\n", x[i]);
	fclose(f);
}
#endif

static void forward(Model * m, uint8_t * image)
{
	Weights *w = &m->weights;
	int dim = m->config.dim;
	int nclass = m->config.nclass;
	Runstate *s = &m->state;
	float *x = s->x;
	QuantizedTensor xq = s->xq;

	normalize(x, image);
	quantize(&xq, x, IMAGE_SZ);
	linear_with_relu(x, &xq, w->wi, w->bi, IMAGE_SZ, dim);
	quantize(&xq, x, dim);
	linear_with_relu(x, &xq, w->wh, w->bh, dim, dim / 2);
	quantize(&xq, x, dim / 2);
	linear(x, &xq, w->wo, w->bo, dim / 2, nclass);
	softmax(x, nclass);
}

static void error_usage()
{
	fprintf(stderr, "Usage:   run <model> <image>\n");
	fprintf(stderr, "Example: run modelq8.bin image1 image2 ... imageN\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{

	char *model_path = NULL;
	char *image_path = NULL;

	// read images and model path, then outputs the probability distribution for the given images.
	if (argc < 3) {
		error_usage();
	}
	model_path = argv[1];
	Model model;
	build_model(&model, model_path);
	uint8_t *image = malloc(IMAGE_SZ);
	for (int i = 2; i < argc; i++) {
		image_path = argv[i];
		read_mnist_image(image_path, image);
		forward(&model, image);	// output (nclass,) is stored in model.state.x2
		for (int j = 0; j < model.config.nclass; j++) {
			printf("%f\t", model.state.x[j]);
		}
		printf("\n");
	}

	free(image);
	free_model(&model);
	return 0;
}
