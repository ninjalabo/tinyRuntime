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

#include "func.h"
#include "func_sq.h"

#define IMAGE_SZ (3 * 224 * 224)	// model input image size (all images are resized to this)
#define IM2COL_MAX_SZ (3 * 49 * 112 * 112)	// maximum array size after im2col
#define IM2COL_SECOND_MAX_SZ (64 * 9 * 56 * 56)	// second maximum array size after im2col
#define OUTPUT_MAX_SZ (64 * 112 * 112)	// maximum output size of layers during forward pass

typedef struct {
	float *x;		// buffer to store the output of a layer (IMAGE_SZ,)
	float *x2;
	UQuantizedTensorSQ xq;	// buffer for quantized arrays
	UQuantizedTensorSQ xq2;
	UQuantizedTensorSQ xq3;
} Runstate;		// the current state in the forward pass

typedef struct {
	ModelConfigSQ model_config;
	ConvConfigSQ *conv_config;	// convolutional layers' config
	LinearConfigSQ *linear_config;	// linear layers' config
	BnConfig *bn_config;	// batchnorm layers' config
	ActivationConfigSQ *activation_config;	// activation config
	int8_t *qparams;	// array of all quantized weights and biases
	float *params;		// array of all non-quantized weights and biases
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

static void malloc_run_state(Runstate * s)
{
	// FIX: fix the memory allocation size
	// FIX: we can probably get rid of one of float array
	s->x = calloc(batch_size * OUTPUT_MAX_SZ, sizeof(float));
	s->x2 = calloc(batch_size * OUTPUT_MAX_SZ, sizeof(float));
	s->xq.q = calloc(batch_size * IM2COL_MAX_SZ, sizeof(uint8_t));
	s->xq2.q = calloc(batch_size * IM2COL_MAX_SZ, sizeof(uint8_t));
	s->xq3.q = calloc(batch_size * IM2COL_MAX_SZ, sizeof(uint8_t));
}

static void free_run_state(Runstate * s)
{
	free(s->x);
	free(s->x2);
	free(s->xq.q);
	free(s->xq2.q);
	free(s->xq3.q);
}

static void read_checkpoint(char *path, ModelConfigSQ * mc, ConvConfigSQ ** cc,
			    LinearConfigSQ ** lc, BnConfig ** bc,
			    ActivationConfigSQ ** ac, int8_t ** qparams,
			    float **params, int *fd, float **data,
			    size_t *file_size)
{
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "Couldn't open file %s\n", path);
		exit(EXIT_FAILURE);
	}
	// read model config
	if (fread(mc, sizeof(ModelConfigSQ), 1, file) != 1) {
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
	*cc = (ConvConfigSQ *) (*data + sizeof(ModelConfigSQ) / sizeof(float));
	*lc = (LinearConfigSQ *) (*cc + mc->nconv);
	*bc = (BnConfig *) (*lc + mc->nlinear);
	*ac = (ActivationConfigSQ *) (*bc + mc->nbn);
	// memory map weights, biases and scaling factors
	*qparams = (int8_t *) (*ac + mc->nactivation);
	*params = (float *) (*qparams + mc->nqparams);
}

static void build_model(Model * m, char *checkpoint_path)
{
	// read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, &m->model_config, &m->conv_config,
			&m->linear_config, &m->bn_config,
			&m->activation_config, &m->qparams, &m->params,
			&m->fd, &m->data, &m->file_size);
}

static void free_model(Model * m)
{
	// close the memory mapping
	if (m->data != MAP_FAILED) {
		munmap(m->data, m->file_size);
	}
	if (m->fd != -1) {
		close(m->fd);
	}
}

#ifdef DEBUG
// sanity check function for writing tensors,
// e.g., it can be used to evaluate values after a specific layer.
static void write_tensor(float *x, int size)
{
	FILE *f = fopen("test.txt", "wb");
	// for (int i = 0; i < size; i++)
	// 	fprintf(f, "%f\n", x[i]);
	fwrite(x, sizeof(float), size, f);
	fclose(f);
}

static void write_qtensor(uint8_t * x, int size)
{
	FILE *f = fopen("testq.txt", "w");
	for (int i = 0; i < size; i++)
		fprintf(f, "%d\n", x[i]);
	fclose(f);
}

#endif

static void basic_block(UQuantizedTensorSQ *xq, UQuantizedTensorSQ *xq2,
			UQuantizedTensorSQ *xq3, float *x, float *x2,
			int8_t *p, ConvConfigSQ *cc, ActivationConfigSQ *ac,
			int *h, int *w, int *i, int *ia, bool downsample)
{
	int h_prev = *h;
	int w_prev = *w;

	im2col_sq(xq3, xq, cc[*i], h, w);
	conv_q(xq, xq3, p, cc[*i], *h, *w);
	relu_q(xq, cc[*i].oc * (*h) * (*w));
	im2col_sq(xq3, xq, cc[*i + 1], h, w);
	conv_q(xq, xq3, p, cc[*i + 1], *h, *w);
	// perform downsampling for skip connection
	if (downsample) {
		im2col_sq(xq3, xq2, cc[*i + 2], &h_prev, &w_prev);
		conv_q(xq2, xq3, p, cc[*i + 2], *h, *w);
	}

	int oc = downsample ? cc[*i + 2].oc : cc[*i + 1].oc;
	int size = oc * (*h) * (*w);
	*i += downsample ? 3 : 2;
	*ia += downsample ? 4 : 3;
	// FIX: is dequantize needed here?
	dequantize(x, xq, size);
	dequantize(x2, xq2, size);
	matadd(x, x2, size);
	quantize(xq, x, ac[*ia].scale, ac[*ia].zero_point, size);
	relu_q(xq, size);
	matcopy_quantized_tensor(xq2, xq, size);
}

static void bottleneck(UQuantizedTensorSQ *xq, UQuantizedTensorSQ *xq2,
		       UQuantizedTensorSQ *xq3, float *x, float *x2,
		       int8_t *p, ConvConfigSQ *cc, ActivationConfigSQ *ac,
		       int *h, int *w, int *i, int *ia, bool downsample)
{
	int h_prev = *h;
	int w_prev = *w;

	im2col_sq(xq3, xq, cc[*i], h, w);
	conv_q(xq, xq3, p, cc[*i], *h, *w);
	relu_q(xq, cc[*i].oc * (*h) * (*w));
	im2col_sq(xq3, xq, cc[*i + 1], h, w);
	conv_q(xq, xq3, p, cc[*i + 1], *h, *w);
	relu_q(xq, cc[*i + 1].oc * (*h) * (*w));
	im2col_sq(xq3, xq, cc[*i + 2], h, w);
	conv_q(xq, xq3, p, cc[*i + 2], *h, *w);
	// perform downsampling for skip connection
	if (downsample) {
		im2col_sq(xq3, xq2, cc[*i + 3], &h_prev, &w_prev);
		conv_q(xq2, xq3, p, cc[*i + 3], *h, *w);
	}

	int oc = downsample ? cc[*i + 3].oc : cc[*i + 2].oc;
	int size = oc * (*h) * (*w);
	*i += downsample ? 4 : 3;
	*ia += downsample ? 5 : 4;
	// FIX: is dequantize needed here?
	dequantize(x, xq, size);
	dequantize(x2, xq2, size);
	matadd(x, x2, size);
	quantize(xq, x, ac[*ia].scale, ac[*ia].zero_point, size);
	relu_q(xq, size);
	matcopy_quantized_tensor(xq2, xq, size);
}

static void sequential_block(UQuantizedTensorSQ *xq, UQuantizedTensorSQ *xq2,
			     UQuantizedTensorSQ *xq3, float *x, float *x2,
			     int8_t *p, ConvConfigSQ *cc,
			     ActivationConfigSQ *ac, int *h, int *w, int *i,
			     int *ia, int n, bool downsample)
{
	if (downsample) {
		basic_block(xq, xq2, xq3, x, x2, p, cc, ac, h, w, i, ia, true);
		for (int j = 0; j < n - 1; j++) {
			basic_block(xq, xq2, xq3, x, x2, p, cc, ac, h, w, i,
				    ia, false);
		}
	} else {
		for (int j = 0; j < n; j++) {
			basic_block(xq, xq2, xq3, x, x2, p, cc, ac, h, w, i,
				    ia, false);
		}
	}
}

static void sequential_bottleneck(UQuantizedTensorSQ *xq,
				  UQuantizedTensorSQ *xq2,
				  UQuantizedTensorSQ *xq3, float *x, float *x2,
				  int8_t *p, ConvConfigSQ *cc,
				  ActivationConfigSQ *ac, int *h, int *w,
				  int *i, int *ia, int n, bool downsample)
{
	if (downsample) {
		bottleneck(xq, xq2, xq3, x, x2, p, cc, ac, h, w, i, ia, true);
		for (int j = 0; j < n - 1; j++) {
			bottleneck(xq, xq2, xq3, x, x2, p, cc, ac, h, w, i, ia,
				   false);
		}
	} else {
		for (int j = 0; j < n; j++) {
			bottleneck(xq, xq2, xq3, x, x2, p, cc, ac, h, w, i, ia,
				   false);
		}
	}
}

static void head(UQuantizedTensorSQ *xq, UQuantizedTensorSQ *xq2,
		 float *images, int8_t *p, ConvConfigSQ *cc,
		 ActivationConfigSQ *ac, int *h, int *w)
{
	quantize(xq, images, ac[0].scale, ac[0].zero_point, IMAGE_SZ);
	im2col_sq(xq2, xq, cc[0], h, w);
	conv_q(xq, xq2, p, cc[0], *h, *w);
	relu_q(xq, cc[0].oc * (*h) * (*w));
	maxpool_q(xq2, xq, h, w, cc[0].oc, 3, 2, 1);
	matcopy_quantized_tensor(xq, xq2, cc[0].oc * (*h) * (*w));
}

static void tail(UQuantizedTensorSQ *xq, UQuantizedTensorSQ *xq2,
		 UQuantizedTensorSQ *xq3, float *x,
		 float *x2, int8_t *p, float *f, ConvConfigSQ *cc,
		 LinearConfigSQ *lc, BnConfig *bc, ActivationConfigSQ *ac,
		 int h, int w, int i, int ia)
{
	int h_prev = h;
	int w_prev = w;

	// FIX: is dequantize needed here?
	dequantize(x, xq2, cc[i - 1].oc * h * w);
	maxpool(x2, x, &h, &w, cc[i - 1].oc, h, 1, 0);
	avgpool(x2 + cc[i - 1].oc, x, &h_prev, &w_prev, cc[i - 1].oc, h_prev,
		1, 0);
	quantize(xq, x2, ac[ia].scale, ac[ia].zero_point, lc[0].in);
	dequantize(x2, xq, lc[0].in);
	quantize(xq, x2, ac[ia + 3].scale, ac[ia + 3].zero_point, lc[0].in);
	dequantize(x2, xq, lc[0].in);
	batchnorm(x, x2, f, bc[0], 1);
	ia += 6; // 2 activation + 2 pool + 1 batchnorm + 1 dropout
	quantize(xq, x, ac[ia].scale, ac[ia].zero_point, lc[0].in);
	linear_q(xq2, xq, p, lc[0]);
	relu_q(xq2, lc[0].out);
	dequantize(x2, xq2, lc[0].out);
	batchnorm(x, x2, f, bc[1], 1);
	ia += 3; // 1 activation + 1 batchnorm + 1 dropout
	quantize(xq, x, ac[ia].scale, ac[ia].zero_point, lc[1].in);
	linear_q(xq2, xq, p, lc[1]);
	dequantize(x, xq2, lc[1].out);
}

static void resnet18(UQuantizedTensorSQ *xq, UQuantizedTensorSQ *xq2,
		     UQuantizedTensorSQ *xq3, float *x,
		     float *x2, float *images, int8_t *p, float *f,
		     ConvConfigSQ *cc, LinearConfigSQ *lc, BnConfig *bc,
		     ActivationConfigSQ *ac, int h, int w)
{
	head(xq, xq2, images, p, cc, ac, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	int ia = 2; // take care of which activation layer to use
	// 2 BasicBlocks, no downsampling
	sequential_block(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia, 2,
			 false);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia, 2,
			 true);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia, 2,
			 true);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia, 2,
			 true);
	// fine tuned layers
	tail(xq, xq2, xq3, x, x2, p, f, cc, lc, bc, ac, h, w, i, ia);
}

static void resnet34(UQuantizedTensorSQ *xq, UQuantizedTensorSQ *xq2,
		     UQuantizedTensorSQ *xq3, float *x,
		     float *x2, float *images, int8_t *p, float *f,
		     ConvConfigSQ *cc, LinearConfigSQ *lc, BnConfig *bc,
		     ActivationConfigSQ *ac, int h, int w)
{
	head(xq, xq2, images, p, cc, ac, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	int ia = 2; // take care of which activation layer to use
	// 3 BasicBlocks, no downsampling
	sequential_block(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia, 3,
			 false);
	// 4 BasicBlocks, with one downsampling block
	sequential_block(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia, 4,
			 true);
	// 6 BasicBlocks, with one downsampling block
	sequential_block(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia, 6,
			 true);
	// 3 BasicBlocks, with one downsampling block
	sequential_block(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia, 3,
			 true);
	// fine tuned layers
	tail(xq, xq2, xq3, x, x2, p, f, cc, lc, bc, ac, h, w, i, ia);
}

static void resnet50(UQuantizedTensorSQ *xq, UQuantizedTensorSQ *xq2,
		     UQuantizedTensorSQ *xq3, float *x,
		     float *x2, float *images, int8_t *p, float *f,
		     ConvConfigSQ *cc, LinearConfigSQ *lc, BnConfig *bc,
		     ActivationConfigSQ *ac, int h, int w)
{
	head(xq, xq2, images, p, cc, ac, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	int ia = 2; // take care of which activation layer to use
	// 3 Bottlenecks, with one downsampling block
	sequential_bottleneck(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia,
			      3, true);
	// 4 Bottlenecks, with one downsampling block
	sequential_bottleneck(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia,
			      4, true);
	// 6 Bottlenecks, with one downsampling block
	sequential_bottleneck(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia,
			      6, true);
	// 3 Bottlenecks, with one downsampling block
	sequential_bottleneck(xq, xq2, xq3, x, x2, p, cc, ac, &h, &w, &i, &ia,
			      3, true);
	// fine tuned layers
	tail(xq, xq2, xq3, x, x2, p, f, cc, lc, bc, ac, h, w, i, ia);
}

static void forward(Model *m, Runstate *s, float *images){
	ConvConfigSQ *cc = m->conv_config;
	LinearConfigSQ *lc = m->linear_config;
	BnConfig *bc = m->bn_config;
	ActivationConfigSQ *ac = m->activation_config;
	int8_t *p = m->qparams;
	float *f = m->params;
	float *x = s->x;
	float *x2 = s->x2;
	UQuantizedTensorSQ xq = s->xq;
	UQuantizedTensorSQ xq2 = s->xq2;
	UQuantizedTensorSQ xq3 = s->xq3;

	int h = 224; // height
	int w = 224; // width

	// select the right model based on the number of convolutional layers
	int nconv = m->model_config.nconv;
	if (nconv == 20)
		resnet18(&xq, &xq2, &xq3, x, x2, images, p, f, cc, lc, bc, ac,
			 h, w);
	else if (nconv == 36)
		resnet34(&xq, &xq2, &xq3, x, x2, images, p, f, cc, lc, bc, ac,
			 h, w);
	else if (nconv == 53)
		resnet50(&xq, &xq2, &xq3, x, x2, images, p, f, cc, lc, bc, ac,
			 h, w);
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
	Model model;
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
		malloc_run_state(&state[i]);
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
			if (i == niter -1 && nimages % batch_size != 0) {
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
