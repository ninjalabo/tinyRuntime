#include <math.h>

#include "properties.h"
#include "func_common.h"
#include <stdio.h>

void quantize(UQuantizedTensorSQ * qx, float *x, float scale, int zero_point,
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

void dequantize(float *x, UQuantizedTensorSQ * qx, int size)
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

static uint8_t vec_dot(uint8_t *q, int zq, int8_t *w,
		     int zw, int32_t b, int zo, float scale_coef, int size)
{
	int32_t ival = 0;
        for (int i = 0; i < size; i++) {
		ival += ((int32_t) q[i] - zq) * ((int32_t) w[i] - zw);
        }
	uint8_t val = dequantize_quantize(ival + b, scale_coef, zo);

	return val;
}

void linear_q(UQuantizedTensorSQ *xout, UQuantizedTensorSQ * x, int8_t * p,
	      LinearConfigSQ lc)
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

void conv_q(UQuantizedTensorSQ *xout, UQuantizedTensorSQ * x, int8_t * p,
	    ConvConfigSQ cc, int height, int width)
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
