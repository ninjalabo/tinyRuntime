#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "config_common.h"
#include "config_vanilla.h"
#include "func_common.h"

// avoid division by zero
#define eps 0.00001f

int batch_size = 1;

// TODO: optimize batch processing in all func*.c modules

void read_imagenette_image(char **paths, float *images, int bs)
{
	int nch = 3, h = 224, w = 224;
	size_t img_sz = nch * h * w;
	for (int b = 0; b < bs; b++) {
		FILE *file = fopen(paths[b], "rb");
		if (!file) {
			fprintf(stderr, "Couldn't open file %s\n", paths[b]);
			exit(EXIT_FAILURE);
		}
		if (fread(&images[b * img_sz], sizeof(float),
		    img_sz, file) != img_sz) {
			fprintf(stderr, "Image read failed\n");
			exit(EXIT_FAILURE);
		}
		fclose(file);
	}
}

// FIX: im2col function may be possible to be optimized
static float im2col_get_pixel_float(float *im, int height, int width, int row,
				    int col, int channel, int pad,
				    int start_idx)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[start_idx + col + width * (row + height * channel)];
}

static uint8_t im2col_get_pixel_uqtensor(UQuantizedTensor *im, int height,
					int width, int row, int col,
					int channel, int pad, int start_idx)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return im->zero_point;
	return im->q[start_idx + col + width * (row + height * channel)];
}

static void im2col_populate_column_float(float *col, float *im, int height,
					 int width, int out_height,
					 int out_width, int ksize, int stride,
					 int pad, int c, int col_size,
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
			    im2col_get_pixel_float(im, height, width, in_row, in_col,
					     chan, pad, im_idx);
		}
	}
}

static void im2col_populate_column_uqtensor(UQuantizedTensor *col,
					    UQuantizedTensor *im, int height,
					    int width, int out_height,
					    int out_width, int ksize,
					    int stride, int pad, int c,
					    int col_size, int col_idx,
					    int im_idx)
{
	int w_offset = c % ksize;
	int h_offset = (c / ksize) % ksize;
	int chan = c / ksize / ksize;
	for (int h = 0; h < out_height; h++) {
		for (int w = 0; w < out_width; w++) {
			int in_row = h_offset + h * stride;
			int in_col = w_offset + w * stride;
			int idx = (h * out_width + w) * col_size + c;
			col->q[col_idx + idx] =
			    im2col_get_pixel_uqtensor(im, height, width, in_row,
						in_col, chan, pad, im_idx);
		}
	}
}

void im2col_generic(void *col, void *im, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad,
		    int type)
{
	// im (nchannels, height, width) -> col (out_height * out_width, col_size)
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;

	im2col_populate_column_fn fn;
	if (type == FLOAT) {
		fn = (im2col_populate_column_fn)im2col_populate_column_float;
	} else if (type == UQTENSOR) {
		fn = (im2col_populate_column_fn)im2col_populate_column_uqtensor;
	} else {
		fprintf(stderr, "Invalid im2col type\n");
		exit(EXIT_FAILURE);
	}

	int col_size = nchannels * ksize * ksize;
	for (int bs = 0; bs < batch_size; bs++) {
		int col_idx = bs * col_size * out_height * out_width;
		int im_idx = bs * nchannels * (*height) * (*width);
		for (int c = 0; c < col_size; c++) {
			fn(col, im, *height, *width, out_height, out_width,
			   ksize, stride, pad, c, col_size, col_idx, im_idx);
		}
	}
	// update current height and width
	*height = out_height;
	*width = out_width;
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

static inline float pool_get_max(float x, float y)
{
	return fmax(x, y);
}

static inline float pool_add(float x, float y)
{
	return x + y;
}

void pool_get_pixel_float(void *xout, int xout_idx, void *x, int height,
			  int width, int ksize, int in_start_row,
			  int in_start_col, pool_operation op, int group)
{
	float *xout_ptr = (float *) xout;
	float *x_ptr = (float *) x;
	float val = 0.0f;
	for (int k = 0; k < ksize * ksize; k++) {
		int in_row = in_start_row + k / ksize;
		int in_col = in_start_col + k % ksize;
		if (in_row >= 0 && in_row < height && in_col >= 0
		    && in_col < width) {
			int x_idx =
			    group * height * width + in_row * width + in_col;
			val = op(x_ptr[x_idx], val);
		}
	}
	xout_ptr[xout_idx] = val;
}

void pool_get_pixel_uint8(void *xout, int xout_idx, void *x, int height,
			  int width, int ksize, int in_start_row,
			  int in_start_col, pool_operation op, int group)
{
	// NOTE: This function performs type conversion to support multiple
	// data types, but this conversion has minimal impact on performance.
	uint8_t *xout_ptr = (uint8_t *) xout;
	uint8_t *x_ptr = (uint8_t *) x;
	float val = 0.0f;
	for (int k = 0; k < ksize * ksize; k++) {
		int in_row = in_start_row + k / ksize;
		int in_col = in_start_col + k % ksize;
		if (in_row >= 0 && in_row < height && in_col >= 0
		    && in_col < width) {
			int x_idx =
			    group * height * width + in_row * width + in_col;
			val = op(x_ptr[x_idx], val);
		}
	}
	xout_ptr[xout_idx] = (uint8_t) val;
}

void pool_generic(void *xout, void *x, int *height, int *width, int nchannels,
		  int ksize, int stride, int pad,
		  pool_get_pixel_fn pool_get_pixel, int op_type,
		  bool concat)
{
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;
	int out_size = out_height * out_width;

	pool_operation op;
	if (op_type == MAX_POOL) {
		op = pool_get_max;
	} else if (op_type == AVG_POOL) {
		op = pool_add;
	} else {
		fprintf(stderr, "Invalid pool operation\n");
		exit(EXIT_FAILURE);
	}

	for (int k = 0; k < batch_size; k++) {
		int xout_idx = k * nchannels * out_size;
		xout_idx = concat ? 2 * xout_idx : xout_idx;
		for (int c = 0; c < nchannels; c++) {
			int group = k * nchannels + c;
			for (int pixel = 0; pixel < out_size; pixel++) {
				int out_row = pixel / out_width;
				int out_col = pixel % out_width;
				int in_start_row = out_row * stride - pad;
				int in_start_col = out_col * stride - pad;

				// current xout index
				int c_xout_idx =
				    xout_idx + c * out_size + pixel;
				pool_get_pixel(xout, c_xout_idx, x, *height,
					       *width, ksize, in_start_row,
					       in_start_col, op, group);
			}
		}
	}
	*height = out_height;
	*width = out_width;
}

void concat_pool(float *xout, float *x, int *height, int *width, int nchannels,
		 int ksize, int stride, int pad)
{
	// maxpool first half of the output and avgpool the second half
	int h_prev = *height;
	int w_prev = *width;
	pool_generic(xout, x, height, width, nchannels, ksize, stride, pad,
		     pool_get_pixel_float, MAX_POOL, true);
	int half_out_size = nchannels * (*height) * (*width);
	pool_generic(&xout[half_out_size], x, &h_prev, &w_prev, nchannels,
		     ksize, stride, pad, pool_get_pixel_float, AVG_POOL, true);
	for (int k = 0; k < batch_size; k++) {
		int xout_idx = k * 2 * half_out_size + half_out_size;
		for (int i = 0; i < half_out_size; i++) {
			xout[xout_idx + i] /= (ksize * ksize);
		}
	}
}

void matadd(float *x, float *y, int size)
{
	for (int i = 0; i < batch_size * size; i++) {
		x[i] = x[i] + y[i];
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

void find_max(int *xout, float *x, int nclasses)
{
	for (int bs = 0; bs < batch_size; bs++) {
		float cmax = 0.0f;
		int max_idx;
		for (int i = 0; i < nclasses; i++) {
			if (cmax < x[bs * nclasses + i]) {
				cmax = x[bs * nclasses + i];
				max_idx = i;
			}
		}
		xout[bs] = max_idx;
	}
}

void matcopy_float(float *xout, float *x, int size) {
	memcpy(xout, x, batch_size * size * sizeof(float));
}

// Unify matcopy functions
void matcopy_uqtensor(UQuantizedTensor *xout, UQuantizedTensor *x, int size) {
	memcpy(xout->q, x->q, batch_size * size * sizeof(uint8_t));
	xout->scale = x->scale;
	xout->zero_point = x->zero_point;
}
