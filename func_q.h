#pragma once

#include "properties.h"

extern void linear_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
                     LinearConfigQ lc);

extern void conv_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
		   ConvConfigQ cc, int height, int width);

extern void quantize(QuantizedTensor * qx, float *x, int n, int gs);

extern void quantize2d(QuantizedTensor * qx, float *x, ConvConfigQ cc,
                       int nrows);