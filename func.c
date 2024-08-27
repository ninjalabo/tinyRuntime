#include <math.h>

#include "config_common.h"
#include "config_vanilla.h"
#include "func_common.h"

void im2col(float *col, float *im, ConvConfig cc, int *height, int *width)
{
	im2col_generic(col, im, height, width, cc.ic, cc.ksize, cc.stride,
		       cc.pad, FLOAT);
}

void maxpool(float *xout, float *x, int *height, int *width, int nchannels,
	     int ksize, int stride, int pad)
{
	pool_generic(xout, x, height, width, nchannels, ksize, stride, pad,
		     pool_get_pixel_float, MAX_POOL, false);
}

void avgpool(float *xout, float *x, int *height, int *width, int nchannels,
	     int ksize, int stride, int pad)
{
	pool_generic(xout, x, height, width, nchannels, ksize, stride, pad,
		     pool_get_pixel_float, AVG_POOL, false);
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

void linear(float *xout, float *x, float *p, LinearConfig lc)
{
	// w(nrows,ncols) @ x (ncols,) + b(nrows,) -> xout (nrows,)
	int ncols = lc.in;
	int nrows = lc.out;

	float *w = p + lc.offset;
	float *b = w + ncols * nrows;
	for (int bs = 0; bs < batch_size; bs++) {
		for (int i = 0; i < nrows; i++) {
			float val = 0.0f;
			for (int j = 0; j < ncols; j++) {
				val += w[i * ncols + j] * x[bs * ncols + j];
			}
			float bias_val = lc.bias ? b[i] : 0.0f;
			xout[bs * nrows + i] = val + bias_val;
		}
	}
}

void conv(float *xout, float *x, float *p, ConvConfig cc, int height, int width)
{
	// w (nchannels,nrows) @ x.T (nrows,ncols) + b(nchannels,) -> xout (nchannels,ncols) <-> xout (nchannels, height, width)
	int nchannels = cc.oc;
	int ncols = height * width;
	int nrows = cc.ic * cc.ksize * cc.ksize;
	float *w = p + cc.offset;
	float *b = w + nchannels * nrows;
	for (int bs = 0; bs < batch_size; bs++) {
		int x_idx = bs * ncols * nrows;
		int xout_idx = bs * nchannels * ncols;
		for (int c = 0; c < nchannels; c++) {
			for (int i = 0; i < ncols; i++) {
				float val = 0.0f;
				for (int j = 0; j < nrows; j++) {
					val +=
					    w[c * nrows + j] *
					    x[x_idx + i * nrows + j];
				}
				float bias_val = cc.bias ? b[c] : 0.0f;
				xout[xout_idx + c * ncols + i] = val + bias_val;
			}
		}
	}
}
