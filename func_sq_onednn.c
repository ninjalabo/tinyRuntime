#include <math.h>
#include <stdlib.h>
#include <dnnl.h>

#include "properties.h"
#include "func_common.h"

// Doc: https://oneapi-src.github.io/oneDNN/

// FIX: unify with func_q.c
// FIX: Allocating memory is not necessary if function receives a pointer
// FIX: matmul by considering batch size
// FIX: optimize transpose
// FIX: modify functions apply to batch size
void quantize(UQuantizedTensorSQ * qx, float *x, float scale, int zero_point,
	      int size)
{
	qx->scale = scale;
	qx->zero_point = zero_point;
	for (int i = 0; i < size; i++) {
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
	for (int i = 0; i < size; i++) {
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

void linear_q(UQuantizedTensorSQ *xout, UQuantizedTensorSQ * x, int8_t * p,
	      LinearConfigSQ lc)
{
	//  x (ncols,) @ T(w(nrows,ncols)+ b(nrows,) -> xout (nrows,)
	int ncols = lc.in;
	int nrows = lc.out;

	int32_t *C = calloc(batch_size * nrows, sizeof(int32_t));
	int8_t *w = p + lc.qoffset;
	int32_t *b;
	if (lc.has_bias)
		b = (int32_t*) (w + nrows * ncols);
	else {
		b = calloc(nrows, sizeof(int32_t));
	}
	// calculate matmul
	dnnl_gemm_u8s8s32('N', 'T', 'R', 1, nrows, ncols, 1.0f, x->q,
				ncols, x->zero_point, w, ncols,
				lc.zero_point, 1.0f, C, nrows, b);
	// dequantize and quantize
	float scale_coef = lc.scale * x->scale / lc.out_scale;
	for (int i = 0; i < nrows; i++) {
		xout->q[i] = dequantize_quantize(C[i], scale_coef, lc.out_zero_point);
	}
	free(C);
	xout->scale = lc.out_scale;
	xout->zero_point = lc.out_zero_point;
}

void conv_q(UQuantizedTensorSQ *xout, UQuantizedTensorSQ * x, int8_t * p,
	    ConvConfigSQ cc, int height, int width)
{
	// x (ncols,nrows) @ T(w (nchannels,nrows)) + b(nchannels,) -> xout (nchannels,ncols) <-> xout (nchannels, height, width)
	int nchannels = cc.oc;
	int ncols = height * width;
	int nrows = cc.ic * cc.ksize * cc.ksize;

	int32_t *C = calloc(batch_size * nchannels * ncols, sizeof(int32_t));
	int8_t *w = p + cc.qoffset;
	int32_t *b;
	if (cc.has_bias)
		b = (int32_t*) (w + nchannels * nrows);
	else {
		b = calloc(nchannels, sizeof(int32_t));
	}
	// FIXME iterate over batch size or calc directly
	dnnl_gemm_u8s8s32('N', 'T', 'R', ncols, nchannels, nrows, 1.0f, x->q, nrows, x->zero_point, w, nrows, cc.zero_point, 1.0f, C, nchannels, b);
	// transpose
	int32_t *Ct = malloc(nchannels * ncols * sizeof(int32_t));
	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < ncols; i++) {
			Ct[c * ncols + i] = C[i * nchannels + c];
		}
	}
	// dequantize
	float scale_coef = cc.scale * x->scale / cc.out_scale;
	for (int i = 0; i < nchannels * ncols; i++) {
		xout->q[i] = dequantize_quantize(Ct[i], scale_coef, cc.out_zero_point);
	}
	free(C);
	free(Ct);
	xout->scale = cc.out_scale;
	xout->zero_point = cc.out_zero_point;
}
