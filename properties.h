
#pragma once

#include <stdint.h>

typedef struct {
	int nclasses;		// number of classes
	int nconv;		// number of convolutional layers
	int nlinear;		// number of linear layers
	int nbn;		// number of batchnorm layers
} ModelConfig;

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

typedef struct {
	int ic;			// input channels
	int offset;		// offset for parameters
} BnConfig;

typedef struct {
	int8_t *q;		// quantized values
	float *s;		// scaling factors
} QuantizedTensor;

typedef struct {
	int nclasses;		// number of classes
	int nconv;		// number of convolutional layers
	int nlinear;		// number of linear layers
	int nbn;		// number of batchnorm layers
	int nparameters;	// number of parameters
} ModelConfigQ;

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
