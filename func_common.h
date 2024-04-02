#pragma once

#include "properties.h"

extern void im2col(float *col, float *im, ConvConfig cc, int *height,
		   int *width);

extern void im2col_q(float *col, float *im, ConvConfigQ cc, int *height,
		     int *width);

extern void batchnorm(float *xout, float *x, float *p, BnConfig bc, int height,
		      int width);

extern void maxpool(float *xout, float *x, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad);

extern void avgpool(float *xout, float *x, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad);

extern void relu(float *x, int size);

extern void softmax(float *x, int size);

// Quantized
extern void linear_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
                     LinearConfigQ lc);

extern void conv_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
		   ConvConfigQ cc, int height, int width);

extern void quantize(QuantizedTensor * qx, float *x, int n, int gs);

extern void quantize2d(QuantizedTensor * qx, float *x, ConvConfigQ cc,
                       int ncols);