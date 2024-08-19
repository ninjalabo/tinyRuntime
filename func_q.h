#pragma once

#include "config_common.h"

#ifdef USE_DQ_FUNC
// functions for dynamic quantization
#include "config_dq.h"

extern void im2col_q(float *col, float *im, ConvConfigQ cc, int *height,
		     int *width);

extern void maxpool_q(float *xout, float *x, int *height, int *width,
		      int nchannels, int ksize, int stride, int pad);

extern void avgpool_q(float *xout, float *x, int *height, int *width,
		      int nchannels, int ksize, int stride, int pad);

extern void relu_q(float *x, int size);

extern void linear_q(float *xout, QuantizedTensor *qx, int8_t * p, float *f,
                     LinearConfigQ lc);

extern void conv_q(float *xout, QuantizedTensor *qx, int8_t * p, float *f,
		   ConvConfigQ cc, int height, int width);

extern void quantize(QuantizedTensor *qx, float *x, int n, int gs);

#else
// functions for static quantization
#include "config_sq.h"

extern void im2col_q(UQuantizedTensor *col, UQuantizedTensor *im,
		     ConvConfigQ cc, int *height, int *width);

extern void maxpool_q(UQuantizedTensor *xout, UQuantizedTensor *x,
		      int *height, int *width, int nchannels, int ksize,
		      int stride, int pad);

extern void relu_q(UQuantizedTensor *x, int size);

extern void quantize(UQuantizedTensor *qx, float *x, float scale,
		    int  zero_point, int size);

extern void dequantize(float *x, UQuantizedTensor *qx, int size);

extern void linear_q(UQuantizedTensor *xout, UQuantizedTensor *x, int8_t *p,
		     LinearConfigQ lc);

extern void conv_q(UQuantizedTensor *xout, UQuantizedTensor *x, int8_t *p,
		   ConvConfigQ cc, int height, int width);

#endif

// FIX: unify function contents in func_sq.c and func_dq.c
extern void head(Runstate *s, float *images, int8_t *p, float *f,
		 ConvConfigQ *cc, ActivationConfigQ *ac, int *h, int *w);

extern void tail(Runstate *s, int8_t *p, float *f, ConvConfigQ *cc,
		 LinearConfigQ *lc, BnConfig *bc, ActivationConfigQ *ac, int h,
		 int w, int i, int ia);

extern void basic_block(Runstate *s, int8_t *p, float *f, ConvConfigQ *cc,
			ActivationConfigQ *ac, int *h, int *w, int *i, int *ia,
			bool downsample);

extern void bottleneck(Runstate *s, int8_t *p, float *f, ConvConfigQ *cc,
		       ActivationConfigQ *ac, int *h, int *w, int *i, int *ia,
		       bool downsample);
