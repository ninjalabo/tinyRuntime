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

#define IMAGE_SZ (3 * 224 * 224)	// model input image size (all images are resized to this)
#define IM2COL_MAX_SZ (3 * 49 * 112 * 112)	// maximum array size after im2col
#define IM2COL_SECOND_MAX_SZ (64 * 9 * 56 * 56)	// second maximum array size after im2col
#define OUTPUT_MAX_SZ (64 * 112 * 112)	// maximum output size of layers during forward pass

typedef struct {
	float *x;		// buffer to store the input
	float *x2;		// buffer to store the output of a layer
	float *x3;		// buffer to store the output of a layer
} Runstate;		// the current state in the forward pass

typedef struct {
	ModelConfig model_config;
	ConvConfig *conv_config;	// convolutional layers' config
	LinearConfig *linear_config;	// linear layers' config
	BnConfig *bn_config;		// batchnorm layers' config
	float *parameters;	// array of all weigths and biases
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

static void malloc_run_state(Runstate * s)
{
	s->x = calloc(batch_size * IM2COL_MAX_SZ, sizeof(float));
	s->x2 = calloc(batch_size * OUTPUT_MAX_SZ, sizeof(float));
	s->x3 = calloc(batch_size * IM2COL_SECOND_MAX_SZ, sizeof(float));
}

static void free_run_state(Runstate * s)
{
	free(s->x);
	free(s->x2);
	free(s->x3);
}

static void read_checkpoint(char *path, ModelConfig * mc, ConvConfig ** cc,
			    LinearConfig ** lc, BnConfig ** bc,
			    float **parameters, int *fd, float **data,
			    size_t *file_size)
{
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "Couldn't open file %s\n", path);
		exit(EXIT_FAILURE);
	}
	// read model config
	if (fread(mc, sizeof(ModelConfig), 1, file) != 1) {
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
	*cc = (ConvConfig *) (*data + sizeof(ModelConfig) / sizeof(float));
	*lc = (LinearConfig *) (*cc + mc->nconv);
	*bc = (BnConfig *) (*lc + mc->nlinear);
	// memory map weights and biases
	*parameters = (float *) (*bc + mc->nbn);
}

static void build_model(Model * m, char *checkpoint_path)
{
	// read in the Config and the Weights from the checkpoint
	read_checkpoint(checkpoint_path, &m->model_config, &m->conv_config,
			&m->linear_config, &m->bn_config, &m->parameters,
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
// sanity check function for writing tensors, e.g., it can be used to evaluate values after a specific layer.
static void write_tensor(float *x, int size)
{
	FILE *f = fopen("test.txt", "wb");
	// for (int i = 0; i < size; i++)
	// 	fprintf(f, "%f\n", x[i]);
	fwrite(x, sizeof(float), size, f);
	fclose(f);
}
#endif

static void basic_block(float *x, float *x2, float *x3, float *p,
			ConvConfig *cc, int *h, int *w, int *i,
			bool downsample)
{
	int h_prev = *h;
	int w_prev = *w;

	im2col(x3, x, cc[*i], h, w);
	conv(x, x3, p, cc[*i], *h, *w);
	relu(x, cc[*i].oc * (*h) * (*w));
	im2col(x3, x, cc[*i + 1], h, w);
	conv(x, x3, p, cc[*i + 1], *h, *w);
	// perform downsampling for skip connection
	if (downsample) {
		im2col(x3, x2, cc[*i + 2], &h_prev, &w_prev);
		conv(x2, x3, p, cc[*i + 2], *h, *w);
	}

	int oc = downsample ? cc[*i + 2].oc : cc[*i + 1].oc;
	int size = oc * (*h) * (*w);
	*i += downsample ? 3 : 2;
	matadd(x, x2, size);
	relu(x, size);
	matcopy_float(x2, x, size);
}

 static void bottleneck(float *x, float *x2, float *x3, float *p,
			ConvConfig *cc, int *h, int *w, int *i,
			bool downsample)
{
	int h_prev = *h;
	int w_prev = *w;

	im2col(x3, x, cc[*i], h, w);
	conv(x, x3, p, cc[*i], *h, *w);
	relu(x, cc[*i].oc * (*h) * (*w));
	im2col(x3, x, cc[*i + 1], h, w);
	conv(x, x3, p, cc[*i + 1], *h, *w);
	relu(x, cc[*i + 1].oc * (*h) * (*w));
	im2col(x3, x, cc[*i + 2], h, w);
	conv(x, x3, p, cc[*i + 2], *h, *w);
	// perform downsampling for skip connection
	if (downsample) {
		im2col(x3, x2, cc[*i + 3], &h_prev, &w_prev);
		conv(x2, x3, p, cc[*i + 3], *h, *w);
	}

	int oc = downsample ? cc[*i + 3].oc : cc[*i + 2].oc;
	int size = oc * (*h) * (*w);
	*i += downsample ? 4 : 3;
	matadd(x, x2, size);
	relu(x, size);
	matcopy_float(x2, x, size);
}

 static void sequential_block(float *x, float *x2, float *x3, float *p,
			      ConvConfig *cc, int *h, int *w, int *i, int n,
			      bool downsample)
{
	if (downsample) {
		basic_block(x, x2, x3, p, cc, h, w, i, true);
		for (int j = 0; j < n - 1; j++) {
			basic_block(x, x2, x3, p, cc, h, w, i, false);
		}
	} else {
		for (int j = 0; j < n; j++) {
			basic_block(x, x2, x3, p, cc, h, w, i, false);
		}
	}
}

 static void sequential_bottleneck(float *x, float *x2, float *x3, float *p,
				   ConvConfig *cc, int *h, int *w, int *i,
				   int n, bool downsample)
{
	if (downsample) {
		bottleneck(x, x2, x3, p, cc, h, w, i, true);
		for (int j = 0; j < n - 1; j++) {
			bottleneck(x, x2, x3, p, cc, h, w, i, false);
		}
	} else {
		for (int j = 0; j < n; j++) {
			bottleneck(x, x2, x3, p, cc, h, w, i, false);
		}
	}
}

 static void head(float *x, float *x2, float *images, float *p, ConvConfig *cc,
		  int *h, int *w)
{
	im2col(x, images, cc[0], h, w);
	conv(x2, x, p, cc[0], *h, *w);
	relu(x2, cc[0].oc * (*h) * (*w));
	maxpool(x, x2, h, w, cc[0].oc, 3, 2, 1);
	matcopy_float(x2, x, cc[0].oc * (*h) * (*w));
}

 static void tail(float *x, float *x2, float *x3, float *p, ConvConfig *cc,
		  LinearConfig *lc, BnConfig *bc, int h, int w, int i)
{
	concat_pool(x2, x, &h, &w, cc[i - 1].oc, h, 1, 0);
	batchnorm(x, x2, p, bc[0], 1);
	linear(x2, x, p, lc[0]);
	relu(x2, lc[0].out);
	batchnorm(x, x2, p, bc[1], 1);
	linear(x2, x, p, lc[1]);
	matcopy_float(x, x2, lc[1].out);
}

 static void resnet18(float *x, float *x2, float *x3, float *images, float *p,
		      ConvConfig *cc, LinearConfig *lc, BnConfig *bc, int h,
		      int w)
{
	head(x, x2, images, p, cc, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	// 2 BasicBlocks, no downsampling
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 2, false);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 2, true);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 2, true);
	// 2 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 2, true);
	// fine tuned layers
	tail(x, x2, x3, p, cc, lc, bc, h, w, i);
}

 static void resnet34(float *x, float *x2, float *x3, float *images, float *p,
		      ConvConfig *cc, LinearConfig *lc, BnConfig *bc, int h,
		      int w)
{
	head(x, x2, images, p, cc, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	// 3 BasicBlocks, no downsampling
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 3, false);
	// 4 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 4, true);
	// 6 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 6, true);
	// 3 BasicBlocks, with one downsampling block
	sequential_block(x, x2, x3, p, cc, &h, &w, &i, 3, true);
	// fine tuned layers
	tail(x, x2, x3, p, cc, lc, bc, h, w, i);
}

 static void resnet50(float *x, float *x2, float *x3, float *images, float *p,
		      ConvConfig *cc, LinearConfig *lc, BnConfig *bc, int h,
		      int w)
{
	head(x, x2, images, p, cc, &h, &w);
	int i = 1; // take care of which convolutional layer to use
	// 3 Bottlenecks, with one downsampling block
	sequential_bottleneck(x, x2, x3, p, cc, &h, &w, &i, 3, true);
	// 4 Bottlenecks, with one downsampling block
	sequential_bottleneck(x, x2, x3, p, cc, &h, &w, &i, 4, true);
	// 6 Bottlenecks, with one downsampling block
	sequential_bottleneck(x, x2, x3, p, cc, &h, &w, &i, 6, true);
	// 3 Bottlenecks, with one downsampling block
	sequential_bottleneck(x, x2, x3, p, cc, &h, &w, &i, 3, true);
	// fine tuned layers
	tail(x, x2, x3, p, cc, lc, bc, h, w, i);
}

static void forward(Model *m, Runstate *s, float *images)
{
	ConvConfig *cc = m->conv_config;
	LinearConfig *lc = m->linear_config;
	BnConfig *bc = m->bn_config;
	float *p = m->parameters;
	float *x = s->x;
	float *x2 = s->x2;
	float *x3 = s->x3;

	int h = 224; // height
	int w = 224; // width

	// select the right model based on the number of convolutional layers
	int nconv = m->model_config.nconv;
	if (nconv == 20)
		resnet18(x, x2, x3, images, p, cc, lc, bc, h, w);
	else if (nconv == 36)
		resnet34(x, x2, x3, images, p, cc, lc, bc, h, w);
	else if (nconv == 53)
		resnet50(x, x2, x3, images, p, cc, lc, bc, h, w);
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
	fprintf(stderr, "Usage:   run <model> <image>\n");
	fprintf(stderr, "Example: run resnet18.bin img1 img2 ... imgN\n");
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
