# Model File Format for tinyRuntime

To use tinyRuntime, you need to export your machine learning models into a specific binary format. This document explains how to export model files for three different types of tinyRuntime: vanilla, dynamic quantization (DQ), and static quantization (SQ). Export functions are implemented in `export.ipynb/py`.

## Vanilla Model

To export a vanilla model, use the function `export_model(model, file_path="model.bin")`. This function takes a PyTorch model and exports its information to a binary file at the specified `file_path` in a custom format. Prior to exporting, the function performs batch normalization folding. The binary file format is organized as follows:

- **Header**
  - Number of classes, i.e., size of model output (int32)
  - Number of Conv2d layers in the model (int32)
  - Number of FC (Fully Connected) layers in the model (int32)
  - Number of BN (Batch Normalization) layers in the model (int32)

- **Conv2d Configurations** (for each Conv2d layer, in the order they appear)
  - Kernel size (int32)
  - Stride (int32)
  - Padding (int32)
  - Input channels (int32)
  - Output channels (int32)
  - Offset, the position (in bytes) where the parameters of this layer start, relative to the beginning of the parameter section (int32)
  - Has bias, 1 if bias exist else 0 (int32)

- **FC Configurations** (for each FC layer, in the order they appear)
  - Input size (int32)
  - Output size (int32)
  - Offset, the position (in bytes) where the parameters of this layer start, relative to the beginning of the parameter section (int32)
  - Has bias, 1 if bias exist else 0 (int32)

- **BN Configurations** (for each BN layer, in the order they appear)
  - Size of input and output (int32)
  - Offset, the position (in bytes) where the parameters of this layer start, relative to the beginning of the parameter section (int32)

- **Conv2d Parameters** (for each Conv2d layer, in the order they appear)
  - Weigths (size of weights * float32)
  - Biases if layer has it (size of biases * float32)

- **FC Parameters** (for each FC layer, in the order they appear)
  - Weigths (size of weights * float32)
  - Biases if layer has it (size of biases * float32)


- **BN Parameters** (for each BN layer, in the order they appear)
  - Weigths (size * float32)
  - Biases (size * float32)
  - Gamma, used for scaling (size * float32)
  - Beta, used for shifting (size * float32)

## Dynamically Quantized Model

To dynamically quantize a model and export it, use the function `export_model_dq8(model, file_path="modelq8.bin", gs=64, asymmetric=True)`. This function takes a PyTorch model, quantizes it and exports its information to a binary file at the specified `file_path` in a custom format. Prior to exporting, the function performs batch normalization folding. The quantization used is group quantization, which requires specifying the group size (`gs`). This function supports both asymmetric and symmetric quantization. The type of quantization can be specified using `asymmetric` parameter. The binary file format is organized as follows:

- **Header**
  - Number of classes, i.e., size of model output (int32)
  - Number of Conv2d layers in the model (int32)
  - Number of FC (Fully Connected) layers in the model (int32)
  - Number of BN (Batch Normalization) layers in the model (int32)
  - Number of Activations, this is not required for dynamic quantization, but it is included in the file because statically quantized models require it. We include it to maintain a consistent file format for both types of quantized models. (int32)
  - Number of quantized parameters (int32)
  - Asymmetric, 1 if true else false (int32)

- **Conv2d Configurations** (for each Conv2d layer, in the order they appear)
  - Kernel size (int32)
  - Stride (int32)
  - Padding (int32)
  - Input channels (int32)
  - Output channels (int32)
  - Offset for quantized parameters, the position (in bytes) where the quantized parameters of this layer start, relative to the beginning of the quantized parameter section (int32)
  - Offset for scaling factors (and zero points if asymmetric), the position (in bytes) where the floating point parameters of this layer start, relative to the beginning of the floating point parameter section (int32)
  - Group size of weights (int32)
  - Group size of biases, 0 if bias not exist (int32)

- **FC Configurations** (for each FC layer, in the order they appear)
  - Input size (int32)
  - Output size (int32)
  - Offset for quantized parameters, the position (in bytes) where the quantized parameters of this layer start, relative to the beginning of the quantized parameter section (int32)
  - Offset for scaling factors (and zero points if asymmetric), the position (in bytes) where the floating point parameters of this layer start, relative to the beginning of the floating point parameter section (int32)
  - Group size of weights (int32)
  - Group size of biases, 0 if bias not exist (int32)

NOTE: Unlike Conv2d and FC, BN is not quantized because it doesn't give much benefit
- **BN Configurations** (for each BN layer, in the order they appear)
  - Size of input and output (int32)
  - Offset, the position (in bytes) where the parameters of this layer start, relative to the beginning of the floating point parameter section (int32)

Quantized parameters:
- **Conv2d Quantized Parameters** (for each Conv2d layer, in the order they appear)
  - Weigths (size of weights * int8)
  - Biases if layer has it (size of biases * int8)

- **FC Quantized Parameters** (for each FC layer, in the order they appear)
  - Weigths (size of weights * int8)
  - Biases if layer has it (size of biases * int8)

Floating point parameters:
- **Conv2d Quantization Parameters** (for each Conv2d layer, in the order they appear)
  - Scaling factors (size of scales * float32)
  - Zero point if asymmetric (size of zero points * int32)

- **FC Quantization Parameters** (for each FC layer, in the order they appear)
  - Scaling factors (size of scales * float32)
  - Zero point if asymmetric (size of zero points * int32)

- **BN Parameters** (for each BN layer, in the order they appear)
  - Weigths (size * float32)
  - Biases (size * float32)
  - Gamma, used for scaling (size * float32)
  - Beta, used for shifting (size * float32)

## Statically Quantized Model

To export statically quantized model, use the function `export_model_sq8(qmodel, model_prepared, file_path="modelq8.bin")`. Unlike `export_model_dq8`, the model has to be quantized before passing to the function. This function takes a quantized PyTorch model `qmodel` and a non-quantized model which is calibrated to be quantized. The `model_prepared` is needed because it contains necessary information that `qmodel` doesn't have. After receiving these models, the function exports the quantized model to a binary file at the specified `file_path` in a custom format. The quantization used is per tensor quantization, which quantizes all weights in a layer by one scaling factor and zero point. This function only supports asymmetric quantization. The binary file format is organized as follows:

- **Header**
  - Number of classes, i.e., size of model output (int32)
  - Number of Conv2d layers in the model (int32)
  - Number of FC (Fully Connected) layers in the model (int32)
  - Number of BN (Batch Normalization) layers in the model (int32)
  - Number of Activations (int32)
  - Number of quantized parameters (int32)
  - Asymmetric, always 1 (int32)

- **Conv2d Configurations** (for each Conv2d layer, in the order they appear)
  - Kernel size (int32)
  - Stride (int32)
  - Padding (int32)
  - Input channels (int32)
  - Output channels (int32)
  - Offset for quantized parameters, the position (in bytes) where the quantized parameters of this layer start, relative to the beginning of the quantized parameter section (int32)
  - Weight scale (float32)
  - Weight zero point (int32)
  - Scale of output activation (float32)
  - Zero point of output activation (int32)
  - Has bias, 1 if bias exist else 0 (int32)

- **FC Configurations** (for each FC layer, in the order they appear)
  - Input size (int32)
  - Output size (int32)
  - Offset for quantized parameters, the position (in bytes) where the quantized parameters of this layer start, relative to the beginning of the quantized parameter section (int32)
  - Weight scale (float32)
  - Weight zero point (int32)
  - Scale of output activation (float32)
  - Zero point of output activation (int32)
  - Has bias, 1 if bias exist else 0 (int32)

NOTE: Unlike Conv2d and FC, BN is not quantized because it doesn't give much benefit
- **BN Configurations** (for each BN layer, in the order they appear)
  - Size of input and output (int32)
  - Offset, the position (in bytes) where the parameters of this layer start, relative to the beginning of the floating point parameter section (int32)

- **Activations Configurations** (for each activation, in the order they appear)
  - Scale (float32)
  - Zero point (int32)

Quantized parameters:
- **Conv2d Quantized Parameters** (for each Conv2d layer, in the order they appear)
  - Weigths (size of weights * int8)
  - Biases if layer has it (size of biases * int32)

- **FC Quantized Parameters** (for each FC layer, in the order they appear)
  - Weigths (size of weights * int8)
  - Biases if layer has it (size of biases * int32)

Floating point parameters:
- **BN Parameters** (for each BN layer, in the order they appear)
  - Weigths (size * float32)
  - Biases (size * float32)
  - Gamma, used for scaling (size * float32)
  - Beta, used for shifting (size * float32)

