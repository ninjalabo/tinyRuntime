#include <blis/blis.h>

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

	float alpha = 1.0f;
	float beta = 0.0f;
	for (int bs = 0; bs < batch_size; bs++) {
		bli_sgemv(BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE, nrows, ncols,
			  &alpha, w, ncols, 1, &x[bs * ncols], 1, &beta,
			  &xout[bs * nrows], 1);
		if (lc.bias) {
			bli_saddv(BLIS_NO_CONJUGATE, nrows, b, 1,
				  &xout[bs * nrows], 1);
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

	float alpha = 1.0f;
	float beta = 0.0f;
	for (int bs = 0; bs < batch_size; bs++) {
		bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, nchannels,
			  ncols, nrows, &alpha, w, nrows, 1,
			  &x[bs * nrows * ncols], 1, nrows, &beta,
			  &xout[bs * nchannels * ncols], ncols, 1);
		if (cc.bias) {
			for (int c = 0; c < nchannels; c++) {
				bli_saddv(BLIS_NO_CONJUGATE, ncols, &b[c], 0,
					  &xout[bs * nchannels * ncols + c *
						ncols], 1);
			}
		}
	}
}
