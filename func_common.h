#pragma once

#include <stdbool.h>

#include "config_common.h"

#define MAX_POOL 0
#define AVG_POOL 1
#define FLOAT 2
#define UQTENSOR 3

extern int batch_size;

extern void read_imagenette_image(char **paths, float *images, int bs);

typedef void (*im2col_get_pixel_fn)(void *col, int col_idx, void *im,
				    int im_idx, int height, int width,
				    int in_row, int in_col, int channel,
				    int pad);

typedef void (*im2col_populate_column_fn)(void *col, void *im, int height,
					  int width, int out_height,
					  int out_width, int ksize, int stride,
					  int pad, int c, int col_size,
					  int col_idx, int im_idx);

extern void im2col_generic(void *col, void *im, int *height, int *width,
			   int nchannels, int ksize, int stride, int pad,
			   int type);

extern void batchnorm(float *xout, float *x, float *p, BnConfig bc, int size);


typedef float (*pool_operation)(float, float);

typedef void (*pool_get_pixel_fn)(void *, int, void *, int, int, int, int, int,
				  pool_operation, int);

extern void pool_get_pixel_float(void *xout, int xout_idx, void *x, int height,
				 int width, int ksize, int in_start_row,
				 int in_start_col, pool_operation op,
				 int group);

extern void pool_get_pixel_uint8(void *xout, int xout_idx, void *x, int height,
				 int width, int ksize, int in_start_row,
				 int in_start_col, pool_operation op,
				 int group);

extern void pool_generic(void *xout, void *x, int *height, int *width,
			 int nchannels, int ksize, int stride, int pad,
			 pool_get_pixel_fn pool_get_pixel, int op_type,
			 bool concat);

extern void concat_pool(float *xout, float *x, int *height, int *width,
			int nchannels, int ksize, int stride, int pad);

extern void matadd(float *x, float *y, int size);

extern void softmax(float *x, int size);

extern void find_max(int *xout, float *x, int nclasses);

extern void matcopy_float(float *xout, float *x, int size);

extern void matcopy_uqtensor(UQuantizedTensor *xout, UQuantizedTensor *x,
			     int size);
