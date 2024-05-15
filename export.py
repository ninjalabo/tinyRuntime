# Don't edit this file! This was automatically generated from "export.ipynb".

import torch
import struct
import numpy as np
from sympy import divisors
import copy
from fasterai.misc.bn_folding import *

def serialize_fp32(file, tensor):
    ''' Write one fp32 tensor to file that is open in wb mode '''
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, tensor):
    ''' Write one int8 tensor to file that is open in wb mode '''
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
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
