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
#include "func_q.h"

#define IMAGE_SZ (3 * 224 * 224)	// model input image size (all images are resized to this)
#define IM2COL_MAX_SZ (3 * 49 * 112 * 112)	// maximum array size after im2col
#define IM2COL_SECOND_MAX_SZ (64 * 9 * 56 * 56)	// second maximum array size after im2col
#define OUTPUT_MAX_SZ (64 * 112 * 112)	// maximum output size of layers during forward pass

typedef struct {
	float *x;		// buffer to store the output of a layer (IMAGE_SZ,)
	float *x2;
	float *x3;
	QuantizedTensor xq;	// buffer for quantized arrays
} Runstate;		// the current state in the forward pass

typedef struct {
	ModelConfigQ model_config;
	ConvConfigQ *conv_config;	// convolutional layers' config
	LinearConfigQ *linear_config;	// linear layers' config
	BnConfig *bn_config;	// batchnorm layers' config
	int8_t *parameters;	// array of all weigths and biases
	float *scaling_factors;	// array of all scaling factors
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

static void malloc_run_state(Runstate * s, int use_zero_point)
{
	s->x = calloc(batch_size * IM2COL_MAX_SZ, sizeof(float));
	s->x2 = calloc(batch_size * OUTPUT_MAX_SZ, sizeof(float));
	s->x3 = calloc(batch_size * IM2COL_SECOND_MAX_SZ, sizeof(float));
	// FIX: too much memory allocated for `xq.s`. Should be divided by group size
	s->xq = (QuantizedTensor) {
	.q = calloc(batch_size * IM2COL_MAX_SZ, sizeof(int8_t)),.s =
		    calloc(batch_size * IM2COL_MAX_SZ, sizeof(float))};
	if (use_zero_point) {
		s->xq.zp = calloc(batch_size * IM2COL_MAX_SZ, sizeof(int));
	} else {
		s->xq.zp = NULL;
	}
}

static void free_run_state(Runstate * s)
{
	free(s->x);
	free(s->x2);
	free(s->x3);
	free(s->xq.q);
	free(s->xq.s);
}

static void read_checkpoint(char *path, ModelConfigQ * mc, ConvConfigQ ** cc,
			    LinearConfigQ ** lc, BnConfig ** bc,
			    int8_t ** parameters, float **scaling_factors,
			    int *fd, float **data, size_t *file_size)
{
	// The data inside the file should follow the order: ModelConfigQ -> ConvConfigQ -> LinearConfigQ -> parameters (first CNN parameters then FC parameters)
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
	// memory map weights, biases and scaling factors
	*parameters = (int8_t *) (*bc + mc->nbn);
	*scaling_factors = (float *)(*parameters + mc->nparameters);
}

static void build_model(Model * m, char *checkpoint_path)
{
	// read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, &m->model_config, &m->conv_config,
			&m->linear_config, &m->bn_config, &m->parameters,
			&m->scaling_factors, &m->fd, &m->data, &m->file_size);
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
	FILE *f = fopen("test2.txt", "w");
	for (int i = 0; i < size; i++)
		fprintf(f, "%f\n", x[i]);
	fclose(f);
}

static void write_qtensor(int8_t * x, int size)
{
	FILE *f = fopen("testq.txt", "w");
	for (int i = 0; i < size; i++)
		fprintf(f, "%d\n", x[i]);
	fclose(f);
}

void dequantize(QuantizedTensor * qx, float *x, int size, int gs)
{
	for (int i = 0; i < size; i++) {
		x[i] = qx->q[i] * qx->s[i / gs];
	}
}

#endif

static void basic_block(QuantizedTensor *xq, float *x, float *x2, float *x3,
			int8_t *p, float *sf, ConvConfigQ *cc, BnConfig *bc,
			int *h, int *w, int *i, bool downsample)
{
	int h_prev = *h;
	int w_prev = *w;

	im2col_q(x3, x, cc[*i], h, w);
	quantize(xq, x3, cc[*i].ic * cc[*i].ksize * cc[*i].ksize * (*h) * (*w),
		 cc[*i].gs_weight);
	conv_q(x3, xq, p, sf, cc[*i], *h, *w);
	batchnorm(x, x3, sf, bc[*i], (*h) * (*w));
	relu(x, cc[*i].oc * (*h) * (*w));
	im2col_q(x3, x, cc[*i + 1], h, w);
	quantize(xq, x3, cc[*i + 1].ic * cc[*i + 1].ksize * cc[*i + 1].ksize *
		 (*h) * (*w), cc[*i + 1].gs_weight);
	conv_q(x3, xq, p, sf, cc[*i + 1], *h, *w);
	batchnorm(x, x3, sf, bc[*i + 1], (*h) * (*w));
	// perform downsampling for skip connection
	if (downsample) {
		im2col_q(x3, x2, cc[*i + 2], &h_prev, &w_prev);
		quantize(xq, x3, cc[*i + 2].ic * cc[*i + 2].ksize *
			 cc[*i + 2].ksize * (*h) * (*w), cc[*i + 2].gs_weight);
		conv_q(x3, xq, p, sf, cc[*i + 2], *h, *w);
		batchnorm(x2, x3, sf, bc[*i + 2], (*h) * (*w));
	}

	int oc = downsample ? cc[*i + 2].oc : cc[*i + 1].oc;
	int size = oc * (*h) * (*w);
	*i += downsample ? 3 : 2;
	matadd(x, x2, size);
	relu(x, size);
	matcopy_float(x2, x, size);
}

 static void bottleneck(QuantizedTensor *xq, float *x, float *x2, float *x3,
			int8_t *p, float *sf, ConvConfigQ *cc, BnConfig *bc,
			int *h, int *w, int *i, bool downsample)
{
	int h_prev = *h;
	int w_prev = *w;

	im2col_q(x3, x, cc[*i], h, w);
	quantize(xq, x3, cc[*i].ic * cc[*i].ksize * cc[*i].ksize * (*h) * (*w),
		 cc[*i].gs_weight);
	conv_q(x3, xq, p, sf, cc[*i], *h, *w);
	batchnorm(x, x3, sf, bc[*i], (*h) * (*w));
	relu(x, cc[*i].oc * (*h) * (*w));
	im2col_q(x3, x, cc[*i + 1], h, w);
	quantize(xq, x3, cc[*i + 1].ic * cc[*i + 1].ksize * cc[*i + 1].ksize *
		 (*h) * (*w), cc[*i + 1].gs_weight);
	conv_q(x3, xq, p, sf, cc[*i + 1], *h, *w);
	batchnorm(x, x3, sf, bc[*i + 1], (*h) * (*w));
	relu(x, cc[*i + 1].oc * (*h) * (*w));
	im2col_q(x3, x, cc[*i + 2], h, w);
	quantize(xq, x3, cc[*i + 2].ic * cc[*i + 2].ksize * cc[*i + 2].ksize *
		 (*h) * (*w), cc[*i + 2].gs_weight);
	conv_q(x3, xq, p, sf, cc[*i + 2], *h, *w);
	batchnorm(x, x3, sf, bc[*i + 2], (*h) * (*w));
	// perform downsampling for skip connection
	if (downsample) {
		im2col_q(x3, x2, cc[*i + 3], &h_prev, &w_prev);
		quantize(xq, x3, cc[*i + 3].ic * cc[*i + 3].ksize *
			 cc[*i + 3].ksize * (*h) * (*w), cc[*i + 3].gs_weight);
		conv_q(x3, xq, p, sf, cc[*i + 3], *h, *w);
		batchnorm(x2, x3, sf, bc[*i + 3], (*h) * (*w));
	}

	int oc = downsample ? cc[*i + 3].oc : cc[*i + 2].oc;
	int size = oc * (*h) * (*w);
	*i += downsample ? 4 : 3;
	matadd(x, x2, size);
	relu(x, size);
	matcopy_float(x2, x, size);
}

 static void sequential_block(QuantizedTensor *xq, float *x, float *x2,
			      float *x3, int8_t *p, float *sf, ConvConfigQ *cc,
			      BnConfig *bc, int *h, int *w, int *i, int n,
			      bool downsample)
{
	if (downsample) {
		basic_block(xq, x, x2, x3, p, sf, cc, bc, h, w, i, true);
		for (int j = 0; j < n - 1; j++) {
			basic_block(xq, x, x2, x3, p, sf, cc, bc, h, w, i,
				    false);
		}
	} else {
		for (int j = 0; j < n; j++) {
			basic_block(xq, x, x2, x3, p, sf, cc, bc, h, w, i,
				    false);
		}
	}
}

 static void sequential_bottleneck(QuantizedTensor *xq, float *x, float *x2,
				   float *x3, int8_t *p, float *sf,
				   ConvConfigQ *cc, BnConfig *bc, int *h,
				   int *w, int *i, int n, bool downsample)
{
	if (downsample) {
		bottleneck(xq, x, x2, x3, p, sf, cc, bc, h, w, i, true);
		for (int j = 0; j < n - 1; j++) {
			bottleneck(xq, x, x2, x3, p, sf, cc, bc, h, w, i,
				   false);
		}
	} else {
		for (int j = 0; j < n; j++) {
			bottleneck(xq, x, x2, x3, p, sf, cc, bc, h, w, i,
				   false);
		}
	}
}

 static void head(QuantizedTensor *xq, float *x, float *x2, float *images,
		  int8_t *p, float *sf, ConvConfigQ *cc, BnConfig *bc, int *h,
		  int *w)
{
	im2col_q(x, images, cc[0], h, w);
	quantize(xq, x, cc[0].ic * cc[0].ksize * cc[0].ksize * (*h) * (*w),
		 cc[0].gs_weight);
	conv_q(x, xq, p, sf, cc[0], *h, *w);
	batchnorm(x2, x, sf, bc[0], (*h) * (*w));
	relu(x2, cc[0].oc * (*h) * (*w));
	maxpool(x, x2, h, w, cc[0].oc, 3, 2, 1);
	matcopy_float(x2, x, cc[0].oc * (*h) * (*w));
}

 static void tail(QuantizedTensor *xq, float *x, float *x2, float *x3,
		  int8_t *p, float *sf, ConvConfigQ *cc, LinearConfigQ *lc,
		  BnConfig *bc, int h, int w, int i)
{
	concat_pool(x2, x, &h, &w, cc[i - 1].oc, h, 1, 0);
	batchnorm(x, x2, sf, bc[i], 1);
	quantize(xq, x, lc[0].in, lc[0].gs_weight);
	linear_q(x2, xq, p, sf, lc[0]);
	relu(x2, lc[0].out);
	batchnorm(x, x2, sf, bc[i + 1], 1);
	quantize(xq, x, lc[1].in, lc[1].gs_weight);
	linear_q(x2, xq, p, sf, lc[1]);
	matcopy_float(x, x2, lc[1].out);
}

 static void resnet18(QuantizedTensor *xq, float *x, float *x2, float *x3,
		      float *images, int8_t *p, float *sf, ConvConfigQ *cc,
		      LinearConfigQ *lc, BnConfig *bc, int h, int w)
{
	head(xq, x, x2, images, p, sf, cc, bc, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	// 2 BasicBlocks, no downsampling
	sequential_block(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 2, false);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 2, true);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 2, true);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 2, true);
	// fine tuned layers
	tail(xq, x, x2, x3, p, sf, cc, lc, bc, h, w, i);
}

 static void resnet34(QuantizedTensor *xq, float *x, float *x2, float *x3,
		      float *images, int8_t *p, float *sf, ConvConfigQ *cc,
		      LinearConfigQ *lc, BnConfig *bc, int h, int w)
{
	head(xq, x, x2, images, p, sf, cc, bc, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	// 3 BasicBlocks, no downsampling
	sequential_block(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 3, false);
	// 4 BasicBlocks, with one downsampling block
	sequential_block(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 4, true);
	// 6 BasicBlocks, with one downsampling block
	sequential_block(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 6, true);
	// 3 BasicBlocks, with one downsampling block
	sequential_block(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 3, true);
	// fine tuned layers
	tail(xq, x, x2, x3, p, sf, cc, lc, bc, h, w, i);
}

 static void resnet50(QuantizedTensor *xq, float *x, float *x2, float *x3,
		      float *images, int8_t *p, float *sf, ConvConfigQ *cc,
		      LinearConfigQ *lc, BnConfig *bc, int h, int w)
{
	head(xq, x, x2, images, p, sf, cc, bc, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	// 3 Bottlenecks, with one downsampling block
	sequential_bottleneck(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 3,
			      true);
	// 4 Bottlenecks, with one downsampling block
	sequential_bottleneck(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 4,
			      true);
	// 6 Bottlenecks, with one downsampling block
	sequential_bottleneck(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 6,
			      true);
	// 3 Bottlenecks, with one downsampling block
	sequential_bottleneck(xq, x, x2, x3, p, sf, cc, bc, &h, &w, &i, 3,
			      true);
	// fine tuned layers
	tail(xq, x, x2, x3, p, sf, cc, lc, bc, h, w, i);
}

static void forward(Model *m, Runstate *s, float *images)
{
	ConvConfigQ *cc = m->conv_config;
	LinearConfigQ *lc = m->linear_config;
	BnConfig *bc = m->bn_config;
	int8_t *p = m->parameters;
	float *sf = m->scaling_factors;
	float *x = s->x;
	float *x2 = s->x2;
	float *x3 = s->x3;
	QuantizedTensor xq = s->xq;

	int h = 224; // height
	int w = 224; // width

	// select the right model based on the number of convolutional layers
	int nconv = m->model_config.nconv;
	if (nconv == 20)
		resnet18(&xq, x, x2, x3, images, p, sf, cc, lc, bc, h, w);
	else if (nconv == 36)
		resnet34(&xq, x, x2, x3, images, p, sf, cc, lc, bc, h, w);
	else if (nconv == 53)
		resnet50(&xq, x, x2, x3, images, p, sf, cc, lc, bc, h, w);
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
