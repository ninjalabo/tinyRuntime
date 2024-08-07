#pragma once

#include "properties.h"

extern void quantize(UQuantizedTensorSQ * qx, float *x, float scale,
		    int  zero_point, int size);

extern void dequantize(float *x, UQuantizedTensorSQ * qx, int size);

extern void linear_q(UQuantizedTensorSQ *xout, UQuantizedTensorSQ * x,
		     int8_t * p, LinearConfigSQ lc);

extern void conv_q(UQuantizedTensorSQ *xout, UQuantizedTensorSQ * x, int8_t * p,
		   ConvConfigSQ cc, int height, int width);
