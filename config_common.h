
#pragma once

#include <stdint.h>

typedef struct {
	int ic;			// input channels
	int offset;		// offset for parameters
} BnConfig;

typedef struct {
	int8_t *q;		// quantized values
	float *scale;		// scaling factors
	int *zero_point;		// zero points
} QuantizedTensor;

typedef struct {
	uint8_t *q;		// quantized values
	float scale;		// scaling factor
	int zero_point;		// zero point
} UQuantizedTensor;
