#include <ctype.h>

#include "config_common.h"

typedef struct {
	float *x;		// buffer to store the output of a layer (IMAGE_SZ,)
	float *x2;
	float *x3;
	QuantizedTensor xq;	// buffer for quantized arrays
} Runstate;		// the current state in the forward pass

typedef struct {
	int ksize;		// kernel size
	int stride;             // kernel stride
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int qoffset;		// offset for quantized parameters
	int soffset;		// offset for scaling factors
	int gs_weight;		// group size for weights
	int gs_bias;		// group size for biases, 0 if layer doesn't have bias
} ConvConfigQ;

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int qoffset;		// offset for quantized parameters
	int soffset;		// offset for scaling factors
	int gs_weight;		// group size for weights
	int gs_bias;		// group size for biases, 0 if layer doesn't have bias
} LinearConfigQ;

typedef struct {
	float scale;		// scaling factor
	int zero_point;		// zero point
} ActivationConfigQ;
