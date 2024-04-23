#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "properties.h"

// avoid division by zero
#define eps 0.00001f

int batch_size = 1;

// TODO: optimize batch processing in all func*.c modules

void read_imagenette_image(char **paths, float *images)
{
	int nch = 3, h = 224, w = 224;
	size_t img_sz = nch * h * w;
	for (int bs = 0; bs < batch_size; bs++) {
		FILE *file = fopen(paths[bs], "rb");
		if (!file) {
			fprintf(stderr, "Couldn't open file %s\n", paths[bs]);
			exit(EXIT_FAILURE);
		}
		if (fread(&images[bs * img_sz], sizeof(float),
		    img_sz, file) != img_sz) {
			fprintf(stderr, "Image read failed\n");
			exit(EXIT_FAILURE);
		}
		fclose(file);
	}
}

static float im2col_get_pixel(float *im, int height, int width, int row,
			      int col, int channel, int pad, int start_idx)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[start_idx + col + width * (row + height * channel)];
}

static void im2col_populate_column(float *col, float *im, int height, int width,
				   int out_height, int out_width, int ksize,
				   int stride, int pad, int c, int col_size,
				   int col_idx, int im_idx)
{
	int w_offset = c % ksize;
	int h_offset = (c / ksize) % ksize;
	int chan = c / ksize / ksize;
	for (int h = 0; h < out_height; h++) {
		for (int w = 0; w < out_width; w++) {
			int in_row = h_offset + h * stride;
			int in_col = w_offset + w * stride;
			int idx = (h * out_width + w) * col_size + c;
			col[col_idx + idx] =
			    im2col_get_pixel(im, height, width, in_row, in_col,
					     chan, pad, im_idx);
		}
	}
}

static void im2col_generic(float *col, float *im, int *height, int *width,
			   int nchannels, int ksize, int stride, int pad)
{
	// im (nchannels, height, width) -> col (out_height * out_width, col_size)
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;

	int col_size = nchannels * ksize * ksize;
	for (int bs = 0; bs < batch_size; bs++) {
		int col_idx = bs * col_size * out_height * out_width;
		int im_idx = bs * nchannels * (*height) * (*width);
		for (int c = 0; c < col_size; c++) {
			im2col_populate_column(col, im, *height, *width,
					       out_height, out_width, ksize,
					       stride, pad, c, col_size,
					       col_idx, im_idx);
		}
	}
	// update current height and width
	*height = out_height;
	*width = out_width;
}

void im2col(float *col, float *im, ConvConfig cc, int *height, int *width)
{
	im2col_generic(col, im, height, width, cc.ic, cc.ksize, cc.stride,
		       cc.pad);
}

void im2col_q(float *col, float *im, ConvConfigQ cc, int *height, int *width)
{
	im2col_generic(col, im, height, width, cc.ic, cc.ksize, cc.stride,
		       cc.pad);
}

void batchnorm_one_group(float *xout, float *x, float *w, float *b,
			 float *rmean, float *rvar, int group, int size,
			 int start_idx)
{
	for (int i = 0; i < size; i++) {
		float val =
		    (x[start_idx + group * size + i] - rmean[group]) /
		    sqrt(rvar[group] + eps) * w[group] + b[group];
		xout[start_idx + group * size + i] = val;
	}
}
// TODO: optimize using blis and/or combine parameters with previous layer
// TODO: make batchnorm1d and batchnorm2d if that is better
void batchnorm(float *xout, float *x, float *p, BnConfig bc, int size)
{
	// x (ngroups,size) -> xout (ngroups,size)
	int ngroups = bc.ic;
	float *w = p + bc.offset;
	float *b = w + ngroups;
	float *rmean = b + ngroups;
	float *rvar = rmean + ngroups;

	for (int bs = 0; bs < batch_size; bs++) {
		int start_idx = bs * ngroups * size;
		for (int group = 0; group < ngroups; group++) {
			batchnorm_one_group(xout, x, w, b, rmean, rvar, group,
					    size, start_idx);
		}
	}
}

typedef float (*PoolOperation)(float, float);

static inline float pool_get_max(float inp, float val)
{
	return fmax(inp, val);
}

static inline float pool_add(float inp, float val)
{
	return val + inp;
}

static float pool_get_pixel(float *x, int height, int width, int ksize,
			    int in_start_row, int in_start_col,
			    PoolOperation op, int group)
{
	float val = 0.0f;
	for (int k = 0; k < ksize * ksize; k++) {
		int in_row = in_start_row + k / ksize;
		int in_col = in_start_col + k % ksize;
		if (in_row >= 0 && in_row < height && in_col >= 0
		    && in_col < width) {
			float inp =
			    x[group * height * width + in_row * width + in_col];
			val = op(inp, val);
		}
	}
	return val;
}

static void pool_generic(float *xout, float *x, int *height, int *width,
			 int nchannels, int ksize, int stride, int pad,
			 PoolOperation op)
{
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;
	int out_size = out_height * out_width;

	for (int group = 0; group < batch_size * nchannels; group++) {
		for (int pixel = 0; pixel < out_size; pixel++) {
			int out_row = pixel / out_width;
			int out_col = pixel % out_width;
			int in_start_row = out_row * stride - pad;
			int in_start_col = out_col * stride - pad;

			float val =
			    pool_get_pixel(x, *height, *width, ksize,
					   in_start_row, in_start_col, op,
					   group);
			xout[group * out_size + pixel] = val;
		}
	}
	*height = out_height;
	*width = out_width;
}

void maxpool(float *xout, float *x, int *height, int *width, int nchannels,
	     int ksize, int stride, int pad)
{
	pool_generic(xout, x, height, width, nchannels, ksize, stride, pad,
		     pool_get_max);
}

void avgpool(float *xout, float *x, int *height, int *width, int nchannels,
	     int ksize, int stride, int pad)
{
	pool_generic(xout, x, height, width, nchannels, ksize, stride, pad,
		     pool_add);
	for (int i = 0; i < batch_size * nchannels * (*height) * (*width); i++)
		xout[i] /= (ksize * ksize);
}

void relu(float *x, int size)
{
	// apply ReLU (Rectified Linear Unit) activation
	for (int i = 0; i < batch_size * size; i++) {
		x[i] = x[i] > 0.0f ? x[i] : 0.0f;
	}
}

void softmax(float *x, int size)
{
	for (int bs = 0; bs < batch_size; bs++) {
		int start_idx = bs * size;
		// find max value (for numerical stability)
		float max_val = x[0];
		for (int i = 1; i < size; i++) {
			if (x[start_idx + i] > max_val) {
				max_val = x[start_idx + i];
			}
		}
		// exp and sum
		float sum = 0.0f;
		for (int i = 0; i < size; i++) {
			x[start_idx + i] = expf(x[start_idx + i] - max_val);
			sum += x[start_idx + i];
		}
		// normalize
		for (int i = 0; i < size; i++) {
			x[start_idx + i] /= sum;
		}
	}
}

void matcopy_float(float *xout, float *x, int size) {
	memcpy(xout, x, batch_size * size * sizeof(float));
}