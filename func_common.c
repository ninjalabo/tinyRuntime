#include <math.h>

#include "properties.h"

// avoid division by zero
#define eps 0.00001f

static float im2col_get_pixel(float *im, int height, int width, int row,
			      int col, int channel, int pad)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[col + width * (row + height * channel)];
}

static void im2col_generic(float *col, float *im, int *height, int *width,
			   int nchannels, int ksize, int stride, int pad)
{
	// im (nchannels, height, width) -> col (out_height * out_width, col_size)
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;

	int col_size = nchannels * ksize * ksize;
	for (int c = 0; c < col_size; c++) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int chan = c / ksize / ksize;
		for (int h = 0; h < out_height; h++) {
			for (int w = 0; w < out_width; w++) {
				int in_row = h_offset + h * stride;
				int in_col = w_offset + w * stride;
				int col_index =
				    (h * out_width + w) * col_size + c;
				col[col_index] =
				    im2col_get_pixel(im, *height, *width,
						     in_row, in_col, chan, pad);
			}
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

// TODO: optimize using blis and/or combine parameters with previous layer
void batchnorm(float *xout, float *x, float *p, BnConfig bc, int height,
	       int width)
{
	// x (nchannels,height,width) -> xout (nchannels,height,width)
	int hw = height * width;
	int nchannels = bc.ic;
	float *w = p + bc.offset;
	float *b = w + nchannels;
	float *rmean = b + nchannels;
	float *rvar = rmean + nchannels;

	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < hw; i++) {
			float val =
			    (x[c * hw + i] - rmean[c]) /
			    sqrt(rvar[c] + eps) * w[c] + b[c];
			xout[c * hw + i] = val;
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
			    PoolOperation op, int c)
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

static void pool_generic(float *xout, float *x, int *height, int *width,
			 int nchannels, int ksize, int stride, int pad,
			 PoolOperation op)
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
			    pool_get_pixel(x, *height, *width, ksize,
					   in_start_row, in_start_col, op, c);
			xout[c * out_size + pixel] = val;
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
	for (int i = 0; i < nchannels * (*height) * (*width); i++) {
		xout[i] /= ksize * ksize;
	}
}

void relu(float *x, int size)
{
	// apply ReLU (Rectified Linear Unit) activation
	for (int i = 0; i < size; i++) {
		x[i] = x[i] > 0.0f ? x[i] : 0.0f;
	}
}

void softmax(float *x, int size)
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
