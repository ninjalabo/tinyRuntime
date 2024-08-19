#pragma once

#include "config_common.h"
#include "config_vanilla.h"

extern void im2col(float *col, float *im, ConvConfig cc, int *height,
		   int *width);

extern void maxpool(float *xout, float *x, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad);

extern void avgpool(float *xout, float *x, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad);

extern void relu(float *x, int size);

extern void linear(float *xout, float *x, float *p, LinearConfig lc);

extern void conv(float *xout, float *x, float *p, ConvConfig cc, int height,
		 int width);
