#pragma once

#include "properties.h"

extern int batch_size;

extern void read_imagenette_image(char **paths, float *images, int bs);

extern void im2col(float *col, float *im, ConvConfig cc, int *height,
		   int *width);

extern void im2col_q(float *col, float *im, ConvConfigQ cc, int *height,
		     int *width);

extern void im2col_sq(UQuantizedTensorSQ *col, UQuantizedTensorSQ *im, ConvConfigSQ cc, int *height,
		     int *width);

extern void batchnorm(float *xout, float *x, float *p, BnConfig bc, int size);

extern void maxpool(float *xout, float *x, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad);

extern void concat_pool(float *xout, float *x, int *height, int *width,
			int nchannels, int ksize, int stride, int pad);

extern void avgpool(float *xout, float *x, int *height, int *width,
		    int nchannels, int ksize, int stride, int pad);

extern void maxpool_q(UQuantizedTensorSQ *xout, UQuantizedTensorSQ *x,
		      int *height, int *width, int nchannels, int ksize,
		      int stride, int pad);

extern void relu(float *x, int size);

extern void relu_q(UQuantizedTensorSQ *x, int size);

extern void softmax(float *x, int size);

extern void find_max(int *xout, float *x, int nclasses);

extern void matcopy_float(float *xout, float *x, int size);

extern void matcopy_quantized_tensor(UQuantizedTensorSQ *xout,
				     UQuantizedTensorSQ *x, int size);
