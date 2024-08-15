
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
	int *zp;		// zero points
} QuantizedTensor;

typedef struct {
	int nclasses;		// number of classes
	int nconv;		// number of convolutional layers
	int nlinear;		// number of linear layers
	int nbn;		// number of batchnorm layers
	int nparameters;	// number of parameters
	int use_zero_point;	// 1 if model uses zero points, 0 otherwise
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

// structs for static quantization with zero points
typedef struct {
	uint8_t *q;		// quantized values
	float scale;		// scaling factor
	int zero_point;		// zero point
} UQuantizedTensorSQ;

typedef struct {
	int nclasses;		// number of classes
	int nconv;		// number of convolutional layers
	int nlinear;		// number of linear layers
	int nbn;		// number of batchnorm layers
	int nactivation;	// number of activation layers
	int nqparams;		// number of quantized parameters
} ModelConfigSQ;

typedef struct {
	int ksize;		// kernel size
	int stride;             // kernel stride
	int pad;		// padding
	int ic;			// input channels
	int oc;			// output channels
	int qoffset;		// offset for quantized parameters
	float scale;		// scaling factor of weights
	int zero_point;		// zero point of weights
	float out_scale;	// scaling factor of layer output
	int out_zero_point;	// zero point of layer output
	int has_bias;		// 1 if layer has bias, 0 otherwise
} ConvConfigSQ;

typedef struct {
	int in;			// input dimension
	int out;		// output dimension
	int qoffset;		// offset for quantized parameters
	float scale;		// scaling factor of weights
	int zero_point;		// zero point of weights
	float out_scale;	// scaling factor of layer output
	int out_zero_point;	// zero point of layer output
	int has_bias;		// 1 if layer has bias, 0 otherwise
} LinearConfigSQ;

typedef struct {
	float scale;		// scaling factor
	int zero_point;		// zero point
} ActivationConfigSQ;
