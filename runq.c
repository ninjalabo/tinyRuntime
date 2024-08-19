/* Copyright (c) 2024 NinjaLABO https://ninjalabo.ai */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

#include "func_q.h"
#include "func_common.h"

#define IMAGE_SZ (3 * 224 * 224)	// model input image size (all images are resized to this)
#define IM2COL_MAX_SZ (3 * 49 * 112 * 112)	// maximum array size after im2col
#define IM2COL_SECOND_MAX_SZ (64 * 9 * 56 * 56)	// second maximum array size after im2col
#define OUTPUT_MAX_SZ (64 * 112 * 112)	// maximum output size of layers during forward pass

typedef struct {
	int nclasses;		// number of classes
	int nconv;		// number of convolutional layers
	int nlinear;		// number of linear layers
	int nbn;		// number of batchnorm layers
	int nactivation;	// number of activation layers
	int nqparams;		// number of quantized parameters
	int use_zero_point;	// 1 if model uses zero points, 0 otherwise
} ModelConfigQ;

typedef struct {
	ModelConfigQ model_config;
	ConvConfigQ *conv_config;	// convolutional layers' config
	LinearConfigQ *linear_config;	// linear layers' config
	BnConfig *bn_config;		// batchnorm layers' config
	ActivationConfigQ *activation_config;	// activation config
	int8_t *qparam_ptr;		// pointer for all quantized weights and biases
	float *param_ptr;		// array of all non-quantized weights and biases
	int fd;				// file descriptor for memory mapping
	float *data;			// memory mapped data pointer
	size_t file_size;		// size of the checkpoint file in bytes
} ModelQ;


static void malloc_run_state(Runstate *s, int use_zero_point)
{
	// FIX: check if the memory size is optimal for the model
	#ifdef USE_DQ_FUNC
	s->x = calloc(batch_size * IM2COL_MAX_SZ, sizeof(float));
	s->x2 = calloc(batch_size * OUTPUT_MAX_SZ, sizeof(float));
	s->x3 = calloc(batch_size * IM2COL_MAX_SZ, sizeof(float));
	s->xq = (QuantizedTensor) {
	    .q = calloc(batch_size * IM2COL_MAX_SZ, sizeof(int8_t)),
	    .scale = calloc(batch_size * IM2COL_MAX_SZ, sizeof(float))};
	if (use_zero_point) {
		s->xq.zero_point =
		    calloc(batch_size * IM2COL_MAX_SZ, sizeof(int));
	} else {
		s->xq.zero_point = NULL;
	}
	#else
	s->x = calloc(batch_size * OUTPUT_MAX_SZ, sizeof(float));
	s->x2 = calloc(batch_size * OUTPUT_MAX_SZ, sizeof(float));
	s->xq.q = calloc(batch_size * IM2COL_MAX_SZ, sizeof(uint8_t));
	s->xq2.q = calloc(batch_size * IM2COL_MAX_SZ, sizeof(uint8_t));
	s->xq3.q = calloc(batch_size * IM2COL_MAX_SZ, sizeof(uint8_t));
	#endif
}

static void free_run_state(Runstate *s)
{
	free(s->x);
	free(s->x2);
	free(s->xq.q);
	#ifdef USE_DQ_FUNC
	free(s->x3);
	free(s->xq.scale);
	#else
	free(s->xq2.q);
	free(s->xq3.q);
	#endif
}

static void read_checkpoint(char *path, ModelConfigQ *mc, ConvConfigQ **cc,
			    LinearConfigQ **lc, BnConfig **bc,
			    ActivationConfigQ **ac, int8_t **qparams,
			    float **params, int *fd, float **data,
			    size_t *file_size)
{
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "Couldn't open file %s\n", path);
		exit(EXIT_FAILURE);
	}
	// read model config
	if (fread(mc, sizeof(ModelConfigQ), 1, file) != 1) {
		exit(EXIT_FAILURE);
	}
	// figure out the file size
	fseek(file, 0, SEEK_END);	// move file pointer to end of file
	*file_size = ftell(file);	// get the file size, in bytes
	fclose(file);
	// memory map layers' config
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
	*cc = (ConvConfigQ *) (*data + sizeof(ModelConfigQ) / sizeof(float));
	*lc = (LinearConfigQ *) (*cc + mc->nconv);
	*bc = (BnConfig *) (*lc + mc->nlinear);
	*ac = (ActivationConfigQ *) (*bc + mc->nbn);
	// memory map weights, biases and scaling factors
	*qparams = (int8_t *) (*ac + mc->nactivation);
	*params = (float *) (*qparams + mc->nqparams);
}

static void build_model(ModelQ *m, char *checkpoint_path)
{
	// read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, &m->model_config, &m->conv_config,
			&m->linear_config, &m->bn_config,
			&m->activation_config, &m->qparam_ptr, &m->param_ptr,
			&m->fd, &m->data, &m->file_size);
}

static void free_model(ModelQ *m)
{
	// close the memory mapping
	if (m->data != MAP_FAILED) {
		munmap(m->data, m->file_size);
	}
	if (m->fd != -1) {
		close(m->fd);
	}
}

static void sequential_block(Runstate *s, int8_t *p, float *sf,
			     ConvConfigQ *cc, ActivationConfigQ *ac, int *h,
			     int *w, int *i, int *ia, int n, bool downsample)
{
	basic_block(s, p, sf, cc, ac, h, w, i, ia, downsample);
	for (int j = 0; j < n - 1; j++) {
		basic_block(s, p, sf, cc, ac, h, w, i, ia, false);
	}
}

static void sequential_bottleneck(Runstate *s, int8_t *p, float *sf,
				  ConvConfigQ *cc, ActivationConfigQ *ac,
				  int *h, int *w, int *i, int *ia, int n,
				  bool downsample)
{
	bottleneck(s, p, sf, cc, ac, h, w, i, ia, downsample);
	for (int j = 0; j < n - 1; j++) {
		bottleneck(s, p, sf, cc, ac, h, w, i, ia, false);
	}
}

static void resnet18(Runstate *s, float *images, int8_t *p, float *sf,
		     ConvConfigQ *cc, LinearConfigQ *lc, BnConfig *bc,
		     ActivationConfigQ *ac, int h, int w)
{
	head(s, images, p, sf, cc, ac, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	int ia = 2; // take care of which activation layer to use
	// 2 BasicBlocks, no downsampling
	sequential_block(s, p, sf, cc, ac, &h, &w, &i, &ia, 2, false);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(s, p, sf, cc, ac, &h, &w, &i, &ia, 2, true);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(s, p, sf, cc, ac, &h, &w, &i, &ia, 2, true);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(s, p, sf, cc, ac, &h, &w, &i, &ia, 2, true);
	// fine tuned layers
	tail(s, p, sf, cc, lc, bc, ac, h, w, i, ia);
}

static void resnet34(Runstate *s, float *images, int8_t *p, float *sf,
		     ConvConfigQ *cc, LinearConfigQ *lc, BnConfig *bc,
		     ActivationConfigQ *ac, int h, int w)
{
	head(s, images, p, sf, cc, ac, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	int ia = 2; // take care of which activation layer to use
	// 3 BasicBlocks, no downsampling
	sequential_block(s, p, sf, cc, ac, &h, &w, &i, &ia, 3, false);
	// 4 BasicBlocks, with one downsampling block
	sequential_block(s, p, sf, cc, ac, &h, &w, &i, &ia, 4, true);
	// 6 BasicBlocks, with one downsampling block
	sequential_block(s, p, sf, cc, ac, &h, &w, &i, &ia, 6, true);
	// 3 BasicBlocks, with one downsampling block
	sequential_block(s, p, sf, cc, ac, &h, &w, &i, &ia, 3, true);
	// fine tuned layers
	tail(s, p, sf, cc, lc, bc, ac, h, w, i, ia);
}

static void resnet50(Runstate *s, float *images, int8_t *p, float *sf,
		     ConvConfigQ *cc, LinearConfigQ *lc, BnConfig *bc,
		     ActivationConfigQ *ac, int h, int w)
{
	head(s, images, p, sf, cc, ac, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	int ia = 2; // take care of which activation layer to use
	// 3 Bottlenecks, with one downsampling block
	sequential_bottleneck(s, p, sf, cc, ac, &h, &w, &i, &ia, 3, true);
	// 4 Bottlenecks, with one downsampling block
	sequential_bottleneck(s, p, sf, cc, ac, &h, &w, &i, &ia, 4, true);
	// 6 Bottlenecks, with one downsampling block
	sequential_bottleneck(s, p, sf, cc, ac, &h, &w, &i, &ia, 6, true);
	// 3 Bottlenecks, with one downsampling block
	sequential_bottleneck(s, p, sf, cc, ac, &h, &w, &i, &ia, 3, true);
	// fine tuned layers
	tail(s, p, sf, cc, lc, bc, ac, h, w, i, ia);
}

static void forward(ModelQ *m, Runstate *s, float *images)
{
	ConvConfigQ *cc = m->conv_config;
	LinearConfigQ *lc = m->linear_config;
	BnConfig *bc = m->bn_config;
	ActivationConfigQ *ac = m->activation_config;
	int8_t *p = m->qparam_ptr;
	float *sf = m->param_ptr;

	int h = 224; // height
	int w = 224; // width

	// select the right model based on the number of convolutional layers
	int nconv = m->model_config.nconv;
	if (nconv == 20)
		resnet18(s, images, p, sf, cc, lc, bc, ac, h, w);
	else if (nconv == 36)
		resnet34(s, images, p, sf, cc, lc, bc, ac, h, w);
	else if (nconv == 53)
		resnet50(s, images, p, sf, cc, lc, bc, ac, h, w);
	else {
		fprintf(stderr, "Invalid structure size. Make sure you have the right model.\n");
		exit(EXIT_FAILURE);
	}
}

static void get_labels(int *labels, char **image_paths, int bs) {
	for (int b = 0; b < bs; b++) {
		// move pointer to last occurence of /
		char *start = strrchr(image_paths[b], '/');
		if (isdigit(*(--start))) {
			labels[b] = atoi(start);
		} else {
			fprintf(stderr, "Label not found from path.\n");
			exit(EXIT_FAILURE);
		}
	}
}

static void process_output(int *correct_count, float *x, int *labels,
			   int *preds, int nclasses, int is_test, int bs) {
	if (is_test) {
		softmax(x, nclasses);
		for (int b = 0; b < bs; b++) {
			for (int j = 0; j < nclasses; j++) {
				printf("%f\t", x[b * nclasses + j]);
			}
		printf("\n");
		}
	} else {
		find_max(preds, x, nclasses);
		for (int b = 0; b < bs; b++) {
			*correct_count += labels[b] == preds[b] ? 1 : 0;
		}
	}
}

static void error_usage()
{
	fprintf(stderr, "Usage:   runq <model> <image>\n");
	fprintf(stderr, "Example: runq resnet18q.bin img1 img2 ... imgN\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Note: If you want to run a test, specify 'test' as the second argument.\n");
	fprintf(stderr, "      This will output the class probability distribution for each image,\n");
	fprintf(stderr, "      otherwise, it will output the accuracy along all images.\n\n");
	exit(EXIT_FAILURE);
}


int main(int argc, char **argv)
{
	char *model_path = NULL;
	char **image_paths = NULL;

	// read images and model path, then outputs the probability distribution for the given images.
	if (argc < 3) {
		error_usage();
	}
	// set global var batch size if environmental var BS is defined
	char* bs_env = getenv("BS");
	int bs = (bs_env != NULL) ? atoi(bs_env) : 1;
	if (bs <= 0) {
		fprintf(stderr, "Invalid batch size\n");
		exit(EXIT_FAILURE);
	}
	batch_size = bs;

	// If the second argument is 'test', the program will output the class
	// probability distribution for each image. Otherwise, it will output
	// the accuracy along all images.
	int is_test = strcmp(argv[1], "test") == 0;
	int argv_idx = is_test ? 2 : 1;
	// build model
	model_path = argv[argv_idx];
	ModelQ model;
	build_model(&model, model_path);
	// define variables
	image_paths = &argv[argv_idx + 1];
	int nimages = argc - argv_idx - 1;
	int niter = (nimages - 1) / batch_size + 1;
	int nclasses = model.model_config.nclasses;
	float total_correct_count = 0.0;
	// allocate memories based on the number of threads
	int max_threads = omp_get_max_threads();
	int nthreads = max_threads < niter ? max_threads : niter;
	float *images =
	    malloc(nthreads * batch_size * IMAGE_SZ * sizeof(float));
	int *labels = malloc(nthreads * batch_size * sizeof(int));
	int *preds = calloc(nthreads * batch_size, sizeof(int));
	// build RunState required for storing intermediate results
	Runstate state[nthreads];
	#pragma omp parallel for
	for (int i = 0; i < nthreads; i++) {
		malloc_run_state(&state[i], model.model_config.use_zero_point);
	}
	#pragma omp parallel
	{
		// define thread specific variables
		int correct_count = 0;
		int thread = omp_get_thread_num();
		float *thread_images = &images[thread * batch_size * IMAGE_SZ];
		int *thread_labels = &labels[thread * batch_size];
		int *thread_preds = &preds[thread * batch_size];

		#pragma omp for nowait
		for (int i = 0; i < niter; i++) {
			// define batch size for this iteration to handle the last batch
			int bs_iter = batch_size;
			if (i == niter - 1 && nimages % batch_size != 0) {
				bs_iter = nimages % batch_size;
			}
			get_labels(thread_labels, &image_paths[i * batch_size],
				   bs_iter);
			read_imagenette_image(&image_paths[i * batch_size],
					      thread_images, bs_iter);
			forward(&model, &state[thread], thread_images);
			// output (nclass,) is stored in model.state.x
			process_output(&correct_count, state[thread].x,
				       thread_labels, thread_preds, nclasses,
				       is_test, bs_iter);
		}
		#pragma omp critical
		{
			total_correct_count += correct_count;
		}
	}

	if (!is_test) {
		printf("%f\n", 100.0f * (float) total_correct_count / nimages);
	}
	free(labels);
	free(preds);
	free(images);
	free_model(&model);
	for (int i = 0; i < nthreads; i++) {
		free_run_state(&state[i]);
	}
	return 0;
}
