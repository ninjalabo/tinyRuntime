# Don't edit this file! This was automatically generated from "export.ipynb".

import struct
import numpy as np
from sympy import divisors
import copy

from fastai.vision.all import *
from fasterai.misc.bn_folding import *

import torch
from torch.ao.quantization import get_default_qconfig_mapping
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from train import load

def serialize_fp32(file, tensor):
    ''' Write one fp32 tensor to file that is open in wb mode '''
    d = tensor.detach().cpu().reshape(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, tensor):
    ''' Write one int8 tensor to file that is open in wb mode '''
    d = tensor.detach().cpu().reshape(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def serialize_int32(file, tensor):
    ''' Write one int32 tensor to file that is open in wb mode '''
    d = tensor.detach().cpu().reshape(-1).numpy().astype(np.int32)
    b = struct.pack(f'{len(d)}i', *d)
    file.write(b)

def quantize_q80(w, group_size):
    '''
    Take a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    '''
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr

def export_model(model, file_path="model.bin"):
    '''
    Export the quantized model to a file
    The data inside the file follows this order:
    1. The number of: classes, each type of layers and parameters
    2. CNN, FC and BN layers' configuration
    3. CNN, FC and BN layers' parameters
    '''
    # batchnorm folding
    model = BN_Folder().fold(model)
    f = open(file_path, "wb")
    # write model config
    conv_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
    nconv = len(conv_layers)
    # read batchnorm1d layers to which batchnorm folding cannot be applied
    bn_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm1d)]
    nbn = len(bn_layers)
    linear_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    nlinear = len(linear_layers)
    nclasses = 10
    header = struct.pack("4i", nclasses, nconv, nlinear, nbn)
    f.write(header)

    # write layers' config
    offset = 0 # the number of bytes in float32 (i.e. offset 1 = 4 bytes)
    for l in conv_layers:
        bias = 1 if l.bias is not None else 0
        f.write(struct.pack("7i", l.kernel_size[0], l.stride[0], l.padding[0],
                l.in_channels, l.out_channels, offset, bias))
        # set offset to the start of next layer
        t_offset =  l.out_channels * l.in_channels * l.kernel_size[0]**2
        # check if the layer has a bias term and adjust the offset accordingly
        offset += t_offset + bias * l.out_channels  # include biases in the offset if bias is not None

    for l in linear_layers:
        bias = 1 if l.bias is not None else 0
        f.write(struct.pack("4i", l.in_features, l.out_features, offset, bias))
        offset += l.in_features * l.out_features + bias * l.out_features

    for l in bn_layers:
        f.write(struct.pack("2i", l.num_features, offset))
        # set offset to the start of next layer
        offset += 4 * l.num_features # weight, bias, running_mean, running_var

    # write the weights and biases of the model
    for l in [*conv_layers, *linear_layers]:
        for p in l.parameters():
            serialize_fp32(f, p)

    for l in bn_layers:
        for p in [l.weight, l.bias, l.running_mean, l.running_var]:
            serialize_fp32(f, p)

    f.close()
    print(f"wrote {file_path}")

def calculate_groupsize(dim, gs):
    '''
    Change the group size if dimension is smaller, and adjust if dim is not a
    multiple of group size. Otherewise it remains the same.
    '''
    if dim < gs:
        return dim
    elif  dim % gs == 0:
        return gs
    else:
        factors = list(divisors(dim)) # give the factors of number "dim"
        return min(factors, key=lambda x: abs(x - gs)) # find the closest number to group size

def export_modelq8(model, file_path="modelq8.bin", gs=64):
    '''
    Export the quantized model to a file
    The data inside the file follows this order:
    1. The number of: classes, each type of layers and parameters
    2. CNN, FC and BN layers' configuration
    3. CNN and FC layers' quantized parameters
    4. CNN and FC layers' scaling factors
    5. BN layers' parameters
    '''
    f = open(file_path, "wb")
    # write model config
    conv_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
    nconv = len(conv_layers)
    linear_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    nlinear = len(linear_layers)
    bn_layers = [layer for layer in model.modules()
                 if isinstance(layer, torch.nn.BatchNorm1d) or isinstance(layer, torch.nn.BatchNorm2d)]
    nbn = len(bn_layers)
    nclasses = 10
    nparameters = sum(p.numel() for layer in [*conv_layers, *linear_layers] for p in layer.parameters())
    header = struct.pack("5i", nclasses, nconv, nlinear, nbn, nparameters)
    f.write(header)
    # write layers' config
    qoffset = 0 # offset for quantized parameters
    soffset = 0 # offset for scaling factors
    group_sizes = [] # save group sizes of each layer
    for l in conv_layers:
        # calculates group sizes for weights and biases
        gs_weight = calculate_groupsize(l.in_channels * l.kernel_size[0]**2, gs)
        gs_bias = calculate_groupsize(l.out_channels, gs) if l.bias is not None else 0
        group_sizes.append(gs_weight)
        if l.bias is not None:
            group_sizes.append(gs_bias)

        f.write(struct.pack("9i", l.kernel_size[0], l.stride[0], l.padding[0], l.in_channels,
                            l.out_channels, qoffset, soffset, gs_weight, gs_bias))
        # set offsets to the start of next layer
        nweights = l.out_channels * l.in_channels * l.kernel_size[0]**2
        if l.bias is not None:
            qoffset += nweights + l.out_channels
            soffset += nweights // gs_weight + l.out_channels // gs_bias
        else:
            qoffset += nweights
            soffset += nweights // gs_weight

    for l in linear_layers:
        gs_weight = calculate_groupsize(l.in_features, gs)
        gs_bias = calculate_groupsize(l.out_features, gs) if l.bias is not None else 0
        group_sizes.append(gs_weight)
        if l.bias is not None:
            group_sizes.append(gs_bias)

        f.write(struct.pack("6i", l.in_features, l.out_features, qoffset, soffset, gs_weight, gs_bias))

        nweights = l.in_features * l.out_features
        if l.bias is not None:
            qoffset += nweights + l.out_features
            soffset += nweights // gs_weight + l.out_features // gs_bias
        else:
            qoffset += nweights
            soffset += nweights // gs_weight   

    for l in bn_layers:
        f.write(struct.pack("2i", l.num_features, soffset))
        # weight, bias, running_mean, running_var
        soffset += 4 * l.num_features

    # write layers' parameters
    ew = []
    scaling_factors = []
    i = 0
    for l in [*conv_layers, *linear_layers]:
        for p in l.parameters():
            q, s, err = quantize_q80(p, group_sizes[i])
            serialize_int8(f, q) # save the tensor in int8
            scaling_factors.append(s)
            ew.append((err, p.shape))
            i += 1
            print(f"Quantized {tuple(p.shape)} to Q8_0 with max error {err}")

    for s in scaling_factors:
        serialize_fp32(f, s) # save scale factors

    for l in bn_layers:
        for p in [l.weight, l.bias, l.running_mean, l.running_var]:
            serialize_fp32(f, p)

    # print the highest error across all parameters, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")
    f.close()
    print(f"wrote {file_path}")

class Quantizer():
    def __init__(self):
        architecture = platform.machine().lower()
        backend = "qnnpack" if 'arm' in architecture or 'aarch64' in architecture else "x86"
        self.qconfig = get_default_qconfig_mapping(backend)
        torch.backends.quantized.engine = backend

    def quantize(self, model, calibration_dls):
        x, _ = calibration_dls.valid.one_batch()
        model_prepared = prepare_fx(model.eval(), self.qconfig, x)
        with torch.no_grad():
            _ = [model_prepared(xb.to('cpu')) for xb, _ in calibration_dls.valid]

        return model_prepared, convert_fx(model_prepared)

def find_input_activation_indices(module, layers, index=0, indices=[]):
    '''
    Find indices for input activations of given layers in the module.
    Parameters:
        module (nn.Module): Module to be analyzed
        index (int): Index of the current activation
        indices (list): List of indices of the input activations
        layers (list): List of layer names whose input activations are to be found
    '''
    if module.__class__.__name__ not in ["GraphModule", "Module", "HistogramObserver"]:
        if module.__class__.__name__ in layers:
            indices.append(index)
        index += 1
        if  module.__class__.__name__=="AdaptiveAvgPool2d":
            index += 2
        return index, indices

    if module.__class__.__name__=="downsample":
        print(module)

    # Iterate through the direct children of the module
    if module.__class__.__name__ not in ["ConvReLU2d", "LinearReLU"]:
        skip_connection_count = 0 # count number of iterations until reaching downsample
        for name, child in module.named_children():
            if name=="downsample":
                indices.append(index - skip_connection_count)
                index += 2
                break;
            skip_connection_count += 1
            index, indices = find_input_activation_indices(child, layers, index, indices)
    return index, indices

def export_modelq8_static(model, model_prepared, file_path="modelq8.bin"):
    '''
    Export the quantized model to a file
    The data inside the file follows this order:
    1. The model config, such as number of each layer
    2. CNN, FC and BN layers' configuration
    3. Activation config: scale and zero point
    FIXME: Now all activation scales & zero points are saved. Most of them can be took from CNN and FC layers,
    so only saving quantized params for addition operation in skip connections is enough.
    4. quantized CNN and FC weights and biases
    7. Non-quantized BN parameters (BN is not quantized at all)
    '''
    f = open(file_path, "wb")
    # write model configs
    nclasses = 10
    # gather convolutional layers
    conv_layers = [layer for layer in model.modules() if layer._get_name() in
                   {'QuantizedConvReLU2d', 'QuantizedConv2d'}]
    nconv = len(conv_layers)
    # gather linear layers
    linear_layers = [layer for layer in model.modules() if layer._get_name() in
                    {'QuantizedLinear', 'QuantizedLinearReLU'}]
    nlinear = len(linear_layers)
    # gather batchnorm layers
    bn_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm1d)]
    nbn = len(bn_layers)
    # calculate number of activations
    nactivation = 0
    while True:
        if hasattr(model_prepared, f'activation_post_process_{nactivation}'):
            nactivation += 1
        else:
            break
    # calculate number of params (number of biases is multiplied by 4 because biases are int32)
    nparams = 0
    for l in [*conv_layers, *linear_layers]:
        nparams += l.weight().numel() + 4 * l.bias().numel() if l.bias else l.weight().numel()
    header = struct.pack("6i", nclasses, nconv, nlinear, nbn, nactivation, nparams)
    f.write(header)
    # write layers' config
    qoffset = 0 # offset for quantized parameters
    group_sizes = [] # save group sizes of each layer
    for l in conv_layers:
        has_bias = 1 if l.bias else 0
        f.write(struct.pack("6ifif2i", l.kernel_size[0], l.stride[0], l.padding[0], l.in_channels,
                            l.out_channels, qoffset, l.weight().q_scale(), l.weight().q_zero_point(),
                            l.scale, l.zero_point, has_bias))
        # 4 * l.bias().numel() because bias is int32 while weight is int8
        qoffset += l.weight().numel() if l.bias is None else l.weight().numel() + 4 * l.bias().numel()

    for l in linear_layers:
        has_bias = 1 if l.bias else 0
        f.write(struct.pack("3ifif2i", l.in_features, l.out_features, qoffset, l.weight().q_scale(),
                            l.weight().q_zero_point(), l.scale, l.zero_point, has_bias))
        qoffset += l.weight().numel() if l.bias is None else l.weight().numel() + 4 * l.bias().numel()

    foffset = 0 # offset for non-quantized parameters
    for l in bn_layers:
        f.write(struct.pack("2i", l.num_features, foffset))
        # (weight, bias, running_mean, running_var) x size of float (4 bytes)
        foffset += 4 * l.num_features

    # write scaling factor and zero point of activation layers
    for i in range(nactivation):
        attr = getattr(model_prepared, f'activation_post_process_{i}')
        scaling_factor, zero_point = attr.calculate_qparams()
        f.write(struct.pack("fi", scaling_factor, zero_point))

    # write layers' quantized weights and biases
    _, activation_indices = find_input_activation_indices(model_prepared, ["Conv2d", "ConvReLU2d", "Linear", "LinearReLU"])
    for i, l in enumerate([*conv_layers, *linear_layers]):
        serialize_int8(f, l.weight().int_repr()) # save the tensor in int8
        if l.bias:
            # Ref: https://discuss.pytorch.org/t/is-bias-quantized-while-doing-pytorch-static-quantization/146416/5
            activation = getattr(model_prepared, f'activation_post_process_{activation_indices[i]}')
            i_scale = activation.calculate_qparams()[0].item()
            b = torch.quantize_per_tensor(l.bias().detach(), i_scale * l.weight().q_scale(), 0, torch.qint32)
            serialize_int32(f, b.int_repr())

    # write batch norm layers' parameters
    for l in bn_layers:
        for p in [l.weight, l.bias, l.running_mean, l.running_var]:
            serialize_fp32(f, p)

    f.close()
    print(f"wrote {file_path}")
