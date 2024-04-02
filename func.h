#pragma once

#include "properties.h"
#include "func_common.h"

extern void linear(float *xout, float *x, float *p, LinearConfig lc);

extern void conv(float *xout, float *x, float *p, ConvConfig cc, int height,
		 int width);

extern void matadd(float *x, float *y, int size);
