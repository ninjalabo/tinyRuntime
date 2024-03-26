
#pragma once

#include "properties.h"

extern void linear(float *xout, float *x, float *p, LinearConfig lc, bool relu);

extern void linear_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
                     LinearConfigQ lc, bool relu);

extern void im2col(float *col, float *im, int *height, int *width,
		   ConvConfig cc);

extern void im2col_q(float *col, float *im, int *height, int *width,
		     ConvConfigQ cc);

extern void conv(float *xout, float *x, float *p, ConvConfig cc, int out,
		 bool relu);

extern void conv_q(float *xout, QuantizedTensor * x, int8_t * p,
		   float *sf, ConvConfigQ cc, int out, bool relu);

extern void batchnorm(float *xout, float *x, float *p, int nchannels, int in,
                      bool relu);

extern void maxpool(float *xout, float *x, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad);

extern void avgpool(float *xout, float *x, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad);

extern void matadd(float *x, float *y, int size);

extern void relu(float *x, int size);

extern void softmax(float *x, int size);

extern void quantize(QuantizedTensor * qx, float *x, int n, int gs);

extern void quantize2d(QuantizedTensor * qx, float *x, ConvConfigQ cc,
                       int ncols);

extern void dequantize(QuantizedTensor * qx, float *x, int n, int gs);

extern void dequantize2d(QuantizedTensor * qx, float *x, int nrows, int ncols,
                         int gs);
