#include <math.h>

#include "config_common.h"
#include "config_sq.h"
#include "func_common.h"

// write unit tests for the following functions
void im2col_q(UQuantizedTensor *col, UQuantizedTensor *im, ConvConfigQ cc,
	      int *height, int *width)
{
	im2col_generic(col, im, height, width, cc.ic, cc.ksize, cc.stride,
		       cc.pad, UQTENSOR);
	col->scale = im->scale;
	col->zero_point = im->zero_point;
}

void maxpool_q(UQuantizedTensor *xout, UQuantizedTensor *x, int *height,
	       int *width, int nchannels, int ksize, int stride, int pad)
{
	pool_generic(xout->q, x->q, height, width, nchannels, ksize, stride,
		     pad, pool_get_pixel_uint8, MAX_POOL, false);
	xout->scale = x->scale;
	xout->zero_point = x->zero_point;
}

void relu_q(UQuantizedTensor *x, int size)
{
	// apply ReLU (Rectified Linear Unit) activation
	for (int i = 0; i < batch_size * size; i++) {
		x->q[i] = x->q[i] > x->zero_point ? x->q[i] : x->zero_point;
	}
}

void quantize(UQuantizedTensor *qx, float *x, float scale, int zero_point,
	      int size)
{
	qx->scale = scale;
	qx->zero_point = zero_point;
	for (int i = 0; i < batch_size * size; i++) {
		int16_t quant_value = (int16_t) roundf(x[i] / qx->scale);
		int16_t quantized = quant_value + qx->zero_point;
		if (quantized > 255) {
			quantized = 255;
		} else if (quantized < 0) {
			quantized = 0;
		}
		qx->q[i] = quantized;
	}
}

void dequantize(float *x, UQuantizedTensor *qx, int size)
{
	for (int i = 0; i < batch_size * size; i++) {
		x[i] = (qx->q[i] - qx->zero_point) * qx->scale;
	}
}

static inline uint8_t dequantize_quantize(int32_t val, float scale_coef,
					  int zero_point)
{
	int16_t quantized = (int16_t) roundf(scale_coef * val) + zero_point;
	// TODO: check if this is necessary
	if (quantized > 255) {
		quantized = 255;
	} else if (quantized < 0) {
		quantized = 0;
	}
	return (uint8_t) quantized;
}

static uint8_t vec_dot(uint8_t *q, int zq, int8_t *w, int zw, int32_t b,
		       int zo, float scale_coef, int size)
{
	int32_t ival = 0;
        for (int i = 0; i < size; i++) {
		ival += ((int32_t) q[i] - zq) * ((int32_t) w[i] - zw);
        }
	uint8_t val = dequantize_quantize(ival + b, scale_coef, zo);

	return val;
}

void linear_q(UQuantizedTensor *xout, UQuantizedTensor *x, int8_t *p,
	      LinearConfigQ lc)
{
	// w(nrows,ncols) @ x (ncols,) + b(nrows,) -> xout (nrows,)
	int ncols = lc.in;
	int nrows = lc.out;

	int8_t *w = p + lc.qoffset;
	int32_t *b = (int32_t*) (w + nrows * ncols);
	float scale_coef = lc.scale * x->scale / lc.out_scale;
	for (int bs = 0; bs < batch_size; bs++) {
		for (int i = 0; i < nrows; i++) {
			int32_t bval = lc.has_bias ? b[i] : 0;
			xout->q[bs * nrows + i] =
			    vec_dot(&x->q[bs * ncols], x->zero_point,
				    &w[i * ncols], lc.zero_point,
				    bval, lc.out_zero_point, scale_coef,
				    ncols);
		}
	}
	xout->scale = lc.out_scale;
	xout->zero_point = lc.out_zero_point;
}

void conv_q(UQuantizedTensor *xout, UQuantizedTensor *x, int8_t *p,
	    ConvConfigQ cc, int height, int width)
{
	// w (nchannels,nrows) @ x (nrows,ncols) + b(nchannels,) -> xout (nchannels,ncols) <-> xout (nchannels, height, width)
	int nchannels = cc.oc;
	int ncols = height * width;
	int nrows = cc.ic * cc.ksize * cc.ksize;

	int8_t *w = p + cc.qoffset;
	int32_t *b = (int32_t*) (w +  nchannels * nrows);
	float scale_coef = cc.scale * x->scale / cc.out_scale;
	for (int bs = 0; bs < batch_size; bs++) {
		int x_idx = bs * nrows * ncols;
		int xout_idx = bs * nchannels * ncols;
		for (int c = 0; c < nchannels; c++) {
			int32_t bval = cc.has_bias ? b[c] : 0;
			for (int i = 0; i < ncols; i++) {
				xout->q[xout_idx + c * ncols + i] =
				    vec_dot(&x->q[x_idx + i * nrows],
				    	    x->zero_point, &w[c * nrows],
					    cc.zero_point, bval,
					    cc.out_zero_point, scale_coef,
					    nrows);
			}
		}
	}
	xout->scale = cc.out_scale;
	xout->zero_point = cc.out_zero_point;
}

// FIX: unify functions below with func_dq.c
void head(Runstate *s, float *images, int8_t *p, float *f, ConvConfigQ *cc,
	  ActivationConfigQ *ac, int *h, int *w)
{
	UQuantizedTensor *xq = &s->xq;
	UQuantizedTensor *xq2 = &s->xq2;

	quantize(xq, images, ac[0].scale, ac[0].zero_point, 3 * (*h) * (*w));
	im2col_q(xq2, xq, cc[0], h, w);
	conv_q(xq, xq2, p, cc[0], *h, *w);
	relu_q(xq, cc[0].oc * (*h) * (*w));
	maxpool_q(xq2, xq, h, w, cc[0].oc, 3, 2, 1);
	matcopy_uqtensor(xq, xq2, cc[0].oc * (*h) * (*w));
}

void tail(Runstate *s, int8_t *p, float *f, ConvConfigQ *cc,
	  LinearConfigQ *lc, BnConfig *bc, ActivationConfigQ *ac, int h, int w,
	  int i, int ia)
{
	float *x = s->x;
	float *x2 = s->x2;
	UQuantizedTensor *xq = &s->xq;
	UQuantizedTensor *xq2 = &s->xq2;
	// FIX: is dequantize needed here?
	dequantize(x, xq2, cc[i - 1].oc * h * w);
	concat_pool(x2, x, &h, &w, cc[i - 1].oc, h, 1, 0);
	quantize(xq, x2, ac[ia].scale, ac[ia].zero_point, lc[0].in);
	dequantize(x2, xq, lc[0].in);
	quantize(xq, x2, ac[ia + 3].scale, ac[ia + 3].zero_point, lc[0].in);
	dequantize(x2, xq, lc[0].in);
	batchnorm(x, x2, f, bc[0], 1);
	ia += 6; // 2 activation + 2 pool + 1 batchnorm + 1 dropout
	quantize(xq, x, ac[ia].scale, ac[ia].zero_point, lc[0].in);
	linear_q(xq2, xq, p, lc[0]);
	relu_q(xq2, lc[0].out);
	dequantize(x2, xq2, lc[0].out);
	batchnorm(x, x2, f, bc[1], 1);
	ia += 3; // 1 activation + 1 batchnorm + 1 dropout
	quantize(xq, x, ac[ia].scale, ac[ia].zero_point, lc[1].in);
	linear_q(xq2, xq, p, lc[1]);
	dequantize(x, xq2, lc[1].out);
}

void basic_block(Runstate *s, int8_t *p, float *f, ConvConfigQ *cc,
		 ActivationConfigQ *ac, int *h, int *w, int *i, int *ia,
		 bool downsample)
{
	float *x = s->x;
	float *x2 = s->x2;
	UQuantizedTensor *xq = &s->xq;
	UQuantizedTensor *xq2 = &s->xq2;
	UQuantizedTensor *xq3 = &s->xq3;

	int h_prev = *h;
	int w_prev = *w;

	im2col_q(xq3, xq, cc[*i], h, w);
	conv_q(xq, xq3, p, cc[*i], *h, *w);
	relu_q(xq, cc[*i].oc * (*h) * (*w));
	im2col_q(xq3, xq, cc[*i + 1], h, w);
	conv_q(xq, xq3, p, cc[*i + 1], *h, *w);
	// perform downsampling for skip connection
	if (downsample) {
		im2col_q(xq3, xq2, cc[*i + 2], &h_prev, &w_prev);
		conv_q(xq2, xq3, p, cc[*i + 2], *h, *w);
	}

	int oc = downsample ? cc[*i + 2].oc : cc[*i + 1].oc;
	int size = oc * (*h) * (*w);
	*i += downsample ? 3 : 2;
	*ia += downsample ? 4 : 3;
	// FIX: is dequantize needed here?
	dequantize(x, xq, size);
	dequantize(x2, xq2, size);
	matadd(x, x2, size);
	quantize(xq, x, ac[*ia].scale, ac[*ia].zero_point, size);
	relu_q(xq, size);
	matcopy_uqtensor(xq2, xq, size);
}

void bottleneck(Runstate *s, int8_t *p, float *f, ConvConfigQ *cc,
		ActivationConfigQ *ac, int *h, int *w, int *i, int *ia,
		bool downsample)
{
	float *x = s->x;
	float *x2 = s->x2;
	UQuantizedTensor *xq = &s->xq;
	UQuantizedTensor *xq2 = &s->xq2;
	UQuantizedTensor *xq3 = &s->xq3;

	int h_prev = *h;
	int w_prev = *w;

	im2col_q(xq3, xq, cc[*i], h, w);
	conv_q(xq, xq3, p, cc[*i], *h, *w);
	relu_q(xq, cc[*i].oc * (*h) * (*w));
	im2col_q(xq3, xq, cc[*i + 1], h, w);
	conv_q(xq, xq3, p, cc[*i + 1], *h, *w);
	relu_q(xq, cc[*i + 1].oc * (*h) * (*w));
	im2col_q(xq3, xq, cc[*i + 2], h, w);
	conv_q(xq, xq3, p, cc[*i + 2], *h, *w);
	// perform downsampling for skip connection
	if (downsample) {
		im2col_q(xq3, xq2, cc[*i + 3], &h_prev, &w_prev);
		conv_q(xq2, xq3, p, cc[*i + 3], *h, *w);
	}

	int oc = downsample ? cc[*i + 3].oc : cc[*i + 2].oc;
	int size = oc * (*h) * (*w);
	*i += downsample ? 4 : 3;
	*ia += downsample ? 5 : 4;
	// FIX: is dequantize needed here?
	dequantize(x, xq, size);
	dequantize(x2, xq2, size);
	matadd(x, x2, size);
	quantize(xq, x, ac[*ia].scale, ac[*ia].zero_point, size);
	relu_q(xq, size);
	matcopy_uqtensor(xq2, xq, size);
}
