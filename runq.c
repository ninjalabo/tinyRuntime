/* Copyright (c) 2024 NinjaLABO https://ninjalabo.ai */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <string.h>

#define IMAGE_SZ (3 * 224 * 224)	// model input image size (all images are resized to this)
#define IM2COL_MAX_SZ (3 * 49 * 112 * 112)	// maximum array size after im2col
#define IM2COL_SECOND_MAX_SZ (64 * 9 * 56 * 56)	// second maximum array size after im2col
#define OUTPUT_MAX_SZ (64 * 112 * 112)	// maximum output size of layers during forward pass

// avoid division by zero
#define eps 0.00001f

typedef struct {
	int8_t *q;		// quantized values
	float *s;		// scaling factors
} QuantizedTensor;

typedef struct {
	int nclasses;		// the number of classes
	int nconv;		// the number of convolutional layers
	int nlinear;		// the number of linear layers
	int nbn;		// the number of batchnorm layers
	int nparameters;	// the number of parameters
} ModelConfig;

typedef struct {
	int ksize;		// kernel size
	int stride;
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int qoffset;		// offset for quantized parameters
	int soffset;		// offset for scaling factors
	int gs_weight;		// group size for weights
	int gs_bias;		// group size for biases, 0 if layer doesn't have bias
} ConvConfig;

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int qoffset;		// offset for quantized parameters
	int soffset;		// offset for scaling factors
	int gs_weight;		// group size for weights
	int gs_bias;		// group size for biases, 0 if layer doesn't have bias
} LinearConfig;

typedef struct {
	int ic;			// input channels
	int offset;		// offset for parameters (stored in scaling_factors as parameters are non-quantized)
} BnConfig;

typedef struct {
	float *x;		// buffer to store the output of a layer (IMAGE_SZ,)
	float *x2;
	float *x3;
	QuantizedTensor xq;	// buffer for quantized arrays
} Runstate;

typedef struct {
	ModelConfig model_config;
	ConvConfig *conv_config;	// convolutional layers' config
	LinearConfig *linear_config;	// linear layers' config
	BnConfig *bn_config;	// batchnorm layers' config
	int8_t *parameters;	// array of all weigths and biases
	float *scaling_factors;	// array of all scaling factors
	Runstate state;		// the current state in the forward pass
	int fd;			// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	size_t file_size;	// size of the checkpoint file in bytes
} Model;

static void malloc_run_state(Runstate * s)
{
	s->x = calloc(IM2COL_MAX_SZ, sizeof(float));
	s->x2 = calloc(OUTPUT_MAX_SZ, sizeof(float));
	s->x3 = calloc(IM2COL_SECOND_MAX_SZ, sizeof(float));

	int gs = 9;		// group size of convolutional layer with the most scaling factors
	s->xq = (QuantizedTensor) {
	.q = calloc(IM2COL_MAX_SZ, sizeof(int8_t)),.s =
		    calloc(IM2COL_SECOND_MAX_SZ / gs, sizeof(float))};
}

static void free_run_state(Runstate * s)
{
	free(s->x);
	free(s->x2);
	free(s->x3);
	free(s->xq.q);
	free(s->xq.s);
}

static void read_checkpoint(char *path, ModelConfig * mc, ConvConfig ** cc,
			    LinearConfig ** lc, BnConfig ** bc,
			    int8_t ** parameters, float **scaling_factors,
			    int *fd, float **data, size_t *file_size)
{
	// The data inside the file should follow the order: ModelConfig -> ConvConfig -> LinearConfig -> parameters (first CNN parameters then FC parameters)
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
	// allocate the RunState buffers
	malloc_run_state(&m->state);
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
	// free the RunState buffers
	free_run_state(&m->state);
}

static void quantize(QuantizedTensor * qx, float *x, int n, int gs)
{
	int num_groups = n / gs;
	float Q_MAX = 127.0f;

	for (int group = 0; group < num_groups; group++) {
		// find the max absolute value in the current group
		float wmax = 0.0;
		for (int i = 0; i < gs; i++) {
			float val = fabs(x[group * gs + i]);
			if (val > wmax) {
				wmax = val;
			}
		}
		// calculate and write the scaling factor
		float scale = wmax / Q_MAX;
		qx->s[group] = scale;

		// calculate and write the quantized values
		for (int i = 0; i < gs; i++) {
			float quant_value = x[group * gs + i] / scale;	// scale
			int8_t quantized = (int8_t) round(quant_value);	// round and clamp
			qx->q[group * gs + i] = quantized;
		}
	}
}

static void quantize2d(QuantizedTensor * qx, float *x, ConvConfig cc, int ncols)
{
	int nrows = cc.ic * cc.ksize * cc.ksize;
	int gs = cc.gs_weight;
	// quantize each column in 2d tensor with a specified group size
	int num_groups = nrows / gs;
	float Q_MAX = 127.0f;

	for (int col = 0; col < ncols; col++) {
		for (int group = 0; group < num_groups; group++) {
			// find the max absolute value in the current group
			float wmax = 0.0;
			for (int i = 0; i < gs; i++) {
				float val =
				    fabs(x[(group * gs + i) * ncols + col]);
				if (val > wmax) {
					wmax = val;
				}
			}
			// calculate and write the scaling factor
			float scale = wmax / Q_MAX;
			qx->s[col * num_groups + group] = scale;

			// calculate and write the quantized values
			for (int i = 0; i < gs; i++) {
				float quant_value = x[(group * gs + i) * ncols + col] / scale;	// scale
				int8_t quantized = (int8_t) round(quant_value);	// round and clamp
				qx->q[(group * gs + i) * ncols + col] =
				    quantized;
			}
		}
	}
}

static void linear(float *xout, QuantizedTensor * x, int8_t * p,
		   float *sf, LinearConfig lc, bool relu)
{
	// w(out,in) @ x (in,) + b(out,) -> xout (out,)
	int in = lc.in;
	int out = lc.out;
	int gs_w = lc.gs_weight;
	int gs_b = lc.gs_bias;

	int i;
	int8_t *w = p + lc.qoffset;
	int8_t *b = w + in * out;
	float *sw = sf + lc.soffset;
	float *sb = sw + in * out / gs_w;
	#pragma omp parallel for private(i)
	for (i = 0; i < out; i++) {
		float val = 0.0f;
		int32_t ival = 0;
		// do the matmul in groups of gs_w
		for (int j = 0; j < in; j += gs_w) {
			for (int k = 0; k < gs_w; k++) {
				ival +=
				    ((int32_t) x->q[j + k]) *
				    ((int32_t) w[i * in + j + k]);
			}
			val +=
			    ((float)ival) * sw[(i * in + j) / gs_w] * x->s[j / gs_w];
			ival = 0;
		}
		float bias_val = gs_b > 0 ? ((float) b[i]) * sb[i / gs_b] : 0.0f;
		xout[i] = relu ? fmax(val + bias_val, 0.0f) : val + bias_val;
	}
}

// linear and convolutional layer operations work only if gs <= in and in % gs = 0
static void matmul_conv(float *xout, QuantizedTensor * x, int8_t * p,
			float *sf, ConvConfig cc, int out, bool relu)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(nchannels,) -> xout (nchannels,out)
	int nchannels = cc.oc;
	int in = cc.ic * cc.ksize * cc.ksize;
	int gs_w = cc.gs_weight;
	int gs_b = cc.gs_bias;

	int c;
	int8_t *w = p + cc.qoffset;
	int8_t *b = w + in * out;
	float *sw = sf + cc.soffset;
	float *sb = sw + in * out / gs_w;
	#pragma omp parallel for private(c)
	for (c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val = 0.0f;
			// do the matmul in groups of gs_w
			for (int j = 0; j < in; j += gs_w) {
				int32_t ival = 0;
				for (int k = 0; k < gs_w; k++) {
					ival +=
					    ((int32_t) x-> q[(j + k) * out + i]) *
					    ((int32_t) w[c * in + (j + k)]);
				}
				val +=
				    (float)ival *x->s[(i * in + j) / gs_w] *
				    sw[(c * in + j) / gs_w];
			}
			float bias_val = gs_b > 0 ? ((float) b[i]) * sb[i / gs_b] : 0.0f;
			xout[c * out + i] =
			    relu ? fmax(val + bias_val, 0.0f) : val + bias_val;
		}
	}
}

static float im2col_get_pixel(float *im, int height, int width, int row,
			      int col, int channel, int pad)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[col + width * (row + height * channel)];
}

static void im2col_cpu(float *col, float *im, int *height, int *width,
		       ConvConfig cc)
{
	// im (nchannels, height, width) -> col (col_size, out_height * out_width)
	int nchannels = cc.ic;
	int ksize = cc.ksize;
	int stride = cc.stride;
	int pad = cc.pad;

	int c, h, w;
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;

	int col_size = nchannels * ksize * ksize;
	#pragma omp parallel for private(c, h, w)
	for (c = 0; c < col_size; c++) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int channel = c / ksize / ksize;
		for (h = 0; h < out_height; h++) {
			for (w = 0; w < out_width; w++) {
				int input_row = h_offset + h * stride;
				int input_col = w_offset + w * stride;
				int col_index =
				    (c * out_height + h) * out_width + w;
				col[col_index] =
				    im2col_get_pixel(im, *height, *width,
						     input_row, input_col,
						     channel, pad);
			}
		}
	}
	// update current height and width
	*height = out_height;
	*width = out_width;
}

static void batchnorm(float *xout, float *x, float *p, int nchannels, int in,
		      bool relu)
{
	// x (nchannels,in) -> xout (nchannels,in)
	float *w = p;
	float *b = p + nchannels;
	float *running_mean = p + 2 * nchannels;
	float *running_var = p + 3 * nchannels;

	int c;
	#pragma omp parallel for private(c)
	for (c = 0; c < nchannels; c++) {
		for (int i = 0; i < in; i++) {
			float val =
			    (x[c * in + i] -
			     running_mean[c]) / sqrt(running_var[c] +
						     eps) * w[c] + b[c];
			xout[c * in + i] = (relu) ? fmax(val, 0.0f) : val;
		}
	}
}

typedef float (*PoolOperation)(float, float);

static inline float get_max(float inp, float val)
{
	return fmax(inp, val);
}

static inline float add(float inp, float val)
{
	return val + inp;
}

static inline float operate_pool(float *x, int height, int width, int ksize,
			   int in_start_row, int in_start_col, PoolOperation op,
			   int c)
{
	float val = 0.0f;
	for (int k = 0; k < ksize * ksize; k++) {
		int in_row = in_start_row + k / ksize;
		int in_col = in_start_col + k % ksize;
		if (in_row >= 0 && in_row < height && in_col >= 0
		    && in_col < width) {
			float inp =
			    x[c * height * width + in_row * width + in_col];
			val = op(inp, val);
		}
	}
	return val;
}

static void pool(float *xout, float *x, int *height, int *width, int nchannels,
		 int ksize, int stride, int pad, PoolOperation op)
{
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;
	int out_size = out_height * out_width;

	for (int c = 0; c < nchannels; c++) {
		for (int pixel = 0; pixel < out_size; pixel++) {
			int out_row = pixel / out_width;
			int out_col = pixel % out_width;
			int in_start_row = out_row * stride - pad;
			int in_start_col = out_col * stride - pad;

			float val =
			    operate_pool(x, *height, *width, ksize,
					 in_start_row, in_start_col, op, c);
			xout[c * out_size + pixel] = val;
		}
	}
	*height = out_height;
	*width = out_width;
}

static void maxpool(float *xout, float *x, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad)
{
	pool(xout, x, height, width, nchannels, ksize, stride, pad, get_max);
}

static void avgpool(float *xout, float *x, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad)
{
	pool(xout, x, height, width, nchannels, ksize, stride, pad, add);
	for (int i = 0; i < nchannels * (*height) * (*width); i++) {
		xout[i] /= ksize * ksize;
	}
}

static void matadd(float *x, float *y, int size)
{
	for (int i = 0; i < size; i++) {
		x[i] = x[i] + y[i];
	}
}

static void relu(float *x, int size)
{
	// apply ReLU (Rectified Linear Unit) activation
	for (int i = 0; i < size; i++) {
		x[i] = fmax(0.0f, x[i]);
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

static void read_imagenette_image(char *path, float **image)
{
	FILE *file = fopen(path, "rb");
	if (!file) {
		fprintf(stderr, "Couldn't open file %s\n", path);
		exit(EXIT_FAILURE);
	}
	if (fread(*image, sizeof(float), IMAGE_SZ, file) != IMAGE_SZ) {
		printf("Image read failed");
		exit(EXIT_FAILURE);
	}
	fclose(file);
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

static void dequantize(QuantizedTensor * qx, float *x, int n, int gs)
{
	for (int i = 0; i < n; i++) {
		x[i] = qx->q[i] * qx->s[i / gs];
	}
}

static void dequantize2d(QuantizedTensor * qx, float *x, int nrows, int ncols,
			 int gs)
{
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			x[i * ncols + j] =
			    qx->q[i * ncols + j] * qx->s[(j * nrows + i) / gs];
		}
	}
}
#endif

static void forward(Model * m, float *image)
{
	ConvConfig *cc = m->conv_config;
	LinearConfig *lc = m->linear_config;
	BnConfig *bc = m->bn_config;
	int8_t *p = m->parameters;
	float *sf = m->scaling_factors;
	Runstate *s = &m->state;
	float *x = s->x;
	float *x2 = s->x2;
	float *x3 = s->x3;
	QuantizedTensor xq = s->xq;

	int h = 224;		// height
	int w = 224;		// width
	int h_prev;		// buffer to store previous height for skip connection
	int w_prev;		// buffer to store previous width for skip connection

	im2col_cpu(x, image, &h, &w, cc[0]);
	quantize2d(&xq, x, cc[0], h * w);
	matmul_conv(x2, &xq, p, sf, cc[0], h * w, false);
	batchnorm(x, x2, sf + bc[0].offset, bc[0].ic, h * w, true);
	maxpool(x2, x, &h, &w, cc[0].oc, 3, 2, 1);
	memcpy(x, x2, cc[0].oc * h * w * sizeof(float));

	// block 1.1 and 1.2
	for (int i = 1; i < 4; i += 2) {
		im2col_cpu(x3, x, &h, &w, cc[i]);
		quantize2d(&xq, x3, cc[i], h * w);
		matmul_conv(x, &xq, p, sf, cc[i], h * w, false);
		batchnorm(x3, x, sf + bc[i].offset, bc[i].ic, h * w, true);
		im2col_cpu(x, x3, &h, &w, cc[i + 1]);
		quantize2d(&xq, x, cc[i + 1], h * w);
		matmul_conv(x3, &xq, p, sf, cc[i + 1], h * w, false);
		batchnorm(x, x3, sf + bc[i + 1].offset, bc[i + 1].ic, h * w,
			  false);
		// skip connection, no change
		matadd(x, x2, cc[i + 1].oc * h * w);
		relu(x, cc[i + 1].oc * h * w);
		memcpy(x2, x, cc[i + 1].oc * h * w * sizeof(float));
	}

	// block 2-4
	for (int i = 5; i < m->model_config.nconv - 4; i += 5) {
		// block i.1
		h_prev = h;
		w_prev = w;
		im2col_cpu(x3, x, &h, &w, cc[i]);
		quantize2d(&xq, x3, cc[i], h * w);
		matmul_conv(x, &xq, p, sf, cc[i], h * w, false);
		batchnorm(x3, x, sf + bc[i].offset, bc[i].ic, h * w, true);
		im2col_cpu(x, x3, &h, &w, cc[i + 1]);
		quantize2d(&xq, x, cc[i + 1], h * w);
		matmul_conv(x3, &xq, p, sf, cc[i + 1], h * w, false);
		batchnorm(x, x3, sf + bc[i + 1].offset, bc[i + 1].ic, h * w,
			  false);
		// skip connection, change in stride
		im2col_cpu(x3, x2, &h_prev, &w_prev, cc[i + 2]);
		quantize2d(&xq, x3, cc[i + 2], h * w);
		matmul_conv(x2, &xq, p, sf, cc[i + 2], h * w, false);
		batchnorm(x3, x2, sf + bc[i + 2].offset, bc[i + 2].ic, h * w,
			  false);
		matadd(x, x3, cc[i + 2].oc * h * w);
		relu(x, cc[i + 2].oc * h * w);
		memcpy(x2, x, cc[i + 2].oc * h * w * sizeof(float));

		// block i.2
		im2col_cpu(x3, x, &h, &w, cc[i + 3]);
		quantize2d(&xq, x3, cc[i + 3], h * w);
		matmul_conv(x, &xq, p, sf, cc[i + 3], h * w, false);
		batchnorm(x3, x, sf + bc[i + 3].offset, bc[i + 3].ic, h * w,
			  true);
		im2col_cpu(x, x3, &h, &w, cc[i + 4]);
		quantize2d(&xq, x, cc[i + 4], h * w);
		matmul_conv(x3, &xq, p, sf, cc[i + 4], h * w, false);
		batchnorm(x, x3, sf + bc[i + 4].offset, bc[i + 4].ic, h * w,
			  false);
		// skip connection, no change
		matadd(x, x2, cc[i + 4].oc * h * w);
		relu(x, cc[i + 4].oc * h * w);
		// the final block output doesn't need to be copied
		if (i < 11) {
			memcpy(x2, x, cc[i + 4].oc * h * w * sizeof(float));
		}
	}

	avgpool(x2, x, &h, &w, cc[m->model_config.nconv - 1].oc, h, 1, 0);
	quantize(&xq, x2, lc[0].in, lc[0].gs_weight);
	linear(x, &xq, p, sf, lc[0], false);
	softmax(x, lc[0].out);
}

static void error_usage()
{
	fprintf(stderr, "Usage:   runq <model> <image>\n");
	fprintf(stderr, "Example: runq modelq8.bin image1 image2 ... imageN\n");
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

	int i;
	#pragma omp parallel for private(i)
	for (i = 2; i < argc; i++) {
		model_path = argv[1];
		Model model;
		build_model(&model, model_path);

		float *image = malloc(IMAGE_SZ * sizeof(float));
		image_path = argv[i];
		read_imagenette_image(image_path, &image);

		forward(&model, image);	// output (nclass,) is stored in model.state.x
		for (int j = 0; j < model.model_config.nclasses; j++) {
			printf("%f\t", model.state.x[j]);
		}
		printf("\n");

		free(image);
		free_model(&model);
	}

	return 0;
}
