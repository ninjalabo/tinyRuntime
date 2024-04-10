#include <blis/blis.h>

#include "properties.h"
#include "func_common.h"

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

void matadd(float *x, float *y, int size)
{
	float alpha = 1.0f;
	bli_saxpyv(BLIS_NO_CONJUGATE, batch_size * size, &alpha, y, 1, x, 1);
}
