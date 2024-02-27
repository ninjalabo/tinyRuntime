# Don't edit this file! This was automatically generated from "export.ipynb".

import torch
import struct
import numpy as np
from sympy import divisors

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
    Export the model to a file
    The data inside the file follows this order:
    1. The number of classes, CNN layers, and FC layers.
    2. CNN and FC layers' configuration.
    3. CNN and FC layers' parameters.
    '''
    f = open(file_path, "wb")
    # write model config
    conv_layers = [model.conv1, model.conv2]
    nconv = len(conv_layers)
    linear_layers = [model.fc1, model.fc2]
    nlinear = len(linear_layers)
    header = struct.pack("iii", model.nclasses, nconv, nlinear)
    f.write(header)
    # write layers' config
    offset = 0 # the number of bytes in float32 (i.e. offset 1 = 4 bytes)
    for layer in conv_layers:
        f.write(struct.pack("6i", layer.kernel_size[0], layer.stride[0], layer.padding[0],
                layer.in_channels, layer.out_channels, offset))
        # set offset to the start of next layer
        offset += layer.out_channels * layer.in_channels * layer.kernel_size[0]**2 + layer.out_channels

    for layer in linear_layers:
        f.write(struct.pack("3i", layer.in_features, layer.out_features, offset))
        offset += layer.in_features * layer.out_features + layer.out_features

    # write the weights and biases of the model
    for layer in conv_layers:
        for p in layer.parameters():
            serialize_fp32(f, p)

    for layer in linear_layers:
        for p in layer.parameters():
            serialize_fp32(f, p)

    f.close()
    print(f"wrote {file_path}")
    
    torch.save(model, "model.pt") # for loading in python

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

def export_modelq8(model_path="model.pt", file_path="modelq8.bin", gs=64):
    '''
    Export the quantized model to a file
    The data inside the file follows this order:
    1. The number of classes, CNN layers, and FC layers, parameters.
    2. CNN and FC layers' configuration.
    3. CNN and FC layers' quantized parameters.
    4. CNN and FC layers' scaling factors.
    '''
    model = torch.load(model_path)
    f = open(file_path, "wb")
    # write model config
    conv_layers = [model.conv1, model.conv2]
    nconv = len(conv_layers)
    linear_layers = [model.fc1, model.fc2]
    nlinear = len(linear_layers)
    nparameters = sum(p.numel() for p in model.parameters())
    header = struct.pack("4i", model.nclasses, nconv, nlinear, nparameters)
    f.write(header)
    # write layers' config
    qoffset = 0 # offset for quantized parameters
    soffset = 0 # offset for scaling factors
    group_sizes = [] # save group sizes of each layer
    for layer in conv_layers:
        # calculates group sizes for weights and biases
        gs_weight = calculate_groupsize(layer.in_channels * layer.kernel_size[0]**2, gs)
        gs_bias = calculate_groupsize(layer.out_channels, gs)
        group_sizes.append(gs_weight)
        group_sizes.append(gs_bias)

        f.write(struct.pack("9i", layer.kernel_size[0], layer.stride[0], layer.padding[0],
                layer.in_channels, layer.out_channels, qoffset, soffset, gs_weight, gs_bias))
        # set offsets to the start of next layer
        nweights = layer.out_channels * layer.in_channels * layer.kernel_size[0]**2
        qoffset += nweights + layer.out_channels
        soffset += nweights // gs_weight + layer.out_channels // gs_bias

    for layer in linear_layers:
        gs_weight = calculate_groupsize(layer.in_features, gs)
        gs_bias = calculate_groupsize(layer.out_features, gs)
        group_sizes.append(gs_weight)
        group_sizes.append(gs_bias)

        f.write(struct.pack("6i", layer.in_features, layer.out_features, qoffset, soffset,
                            gs_weight, gs_bias))

        nweights = layer.in_features * layer.out_features
        qoffset += nweights + layer.out_features
        soffset += nweights // gs_weight + layer.out_features // gs_bias

    ew = []
    scaling_factors = []
    i = 0
    for layer in [*conv_layers, *linear_layers]:
        for p in layer.parameters():
            q, s, err = quantize_q80(p, group_sizes[i])
            serialize_int8(f, q) # save the tensor in int8
            scaling_factors.append(s)
            ew.append((err, p.shape))
            i += 1
            print(f"Quantized {tuple(p.shape)} to Q8_0 with max error {err}")

    for s in scaling_factors:
        serialize_fp32(f, s) # save scale factors

    # print the highest error across all parameters, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")
    f.close()
    print(f"wrote {file_path}")
