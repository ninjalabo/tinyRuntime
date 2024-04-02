#include <blis/blis.h>

#include "properties.h"

void linear(float *xout, float *x, float *p, LinearConfig lc)
{
	// w(nrows,ncols) @ x (ncols,) + b(nrows,) -> xout (nrows,)
	int ncols = lc.in;
	int nrows = lc.out;
	float *w = p + lc.offset;
	float *b = w + ncols * nrows;

	float alpha = 1.0f;
	float beta = 0.0f;
	bli_sgemv(BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE, nrows, ncols, &alpha,
		  w, ncols, 1, x, 1, &beta, xout, 1);
	if (lc.bias) {
		bli_saddv(BLIS_NO_CONJUGATE, nrows, b, 1, xout, 1);
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
	bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, nchannels, ncols, nrows,
		  &alpha, w, nrows, 1, x, 1, nrows, &beta, xout, ncols, 1);
	if (cc.bias) {
		bli_saddm(0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
			  nchannels, ncols, b, 1, 0, xout, ncols, 1);
	}
}

void matadd(float *x, float *y, int size)
{
	float alpha = 1.0f;
	bli_saxpyv(BLIS_NO_CONJUGATE, size, &alpha, y, 1, x, 1);
}
