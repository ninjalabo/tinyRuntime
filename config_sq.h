#include <ctype.h>

#include "config_common.h"

typedef struct {
	float *x;		// buffer to store the output of a layer (IMAGE_SZ,)
	float *x2;
	UQuantizedTensor xq;	// buffer for quantized arrays
	UQuantizedTensor xq2;
	UQuantizedTensor xq3;
} Runstate;		// the current state in the forward pass

typedef struct {
	int ksize;		// kernel size
	int stride;		// kernel stride
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int qoffset;		// offset for quantized parameters
	float scale;		// scaling factor of weights
	int zero_point;		// zero point of weights
	float out_scale;	// scaling factor of layer output
	int out_zero_point;	// zero point of layer output
	int has_bias;		// 1 if layer has bias, 0 otherwise
} ConvConfigQ;

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int qoffset;		// offset for quantized parameters
	float scale;		// scaling factor of weights
	int zero_point;		// zero point of weights
	float out_scale;	// scaling factor of layer output
	int out_zero_point;	// zero point of layer output
	int has_bias;		// 1 if layer has bias, 0 otherwise
} LinearConfigQ;

typedef struct {
	float scale;		// scaling factor
	int zero_point;		// zero point
} ActivationConfigQ;
