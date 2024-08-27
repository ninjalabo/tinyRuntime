#include <ctype.h>

#include "config_common.h"

typedef struct {
	int ksize;		// kernel size
	int stride;             // kernel stride
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int offset;		// offset parameters
	int bias;		// if layer has bias the value equals to 1 else 0
} ConvConfig;

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int offset;		// offset for parameters
	int bias;		// if layer has bias the value equals to 1 else 0
} LinearConfig;
