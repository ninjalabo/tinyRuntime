# Don't edit this file! This was automatically generated from "export.ipynb".

import torch
import struct
import numpy as np

def serialize_fp32(file, tensor):
    """ write one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, tensor):
    """ write one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
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

def export_model(model, file_path = "model.bin"):
    ''' export the model to filepath '''
    f = open(file_path, "wb")
    # write the model structure 
    header = struct.pack("ii", model.dim, model.nclass)
    f.write(header) 
    # write the model weights and biases
    weights = [*[layer.weight for layer in model.layers], model.out.weight]
    bias = [*[layer.bias for layer in model.layers], model.out.bias]
    
    for w in weights:
        serialize_fp32(f, w)
    
    for b in bias:
        serialize_fp32(f, b)

    f.close()
    print(f"wrote {file_path}")
    
    torch.save(model, "model.pt") # for loading in python

def export_modelq8(model_path="model.pt", file_path="modelq8.bin", gs=64):
    ''' read a model from model.bin if not given and export a quatized (int8) model to filepath '''
    model = torch.load(model_path)
    f = open(file_path, "wb")
    # write the model structure 
    header = struct.pack("iii", model.dim, model.nclass, gs)
    f.write(header) 
    # quantize and write the model weights and biases
    weights = [*[layer.weight for layer in model.layers], model.out.weight]
    biases = [*[layer.bias for layer in model.layers], model.out.bias]
    params = [*weights, *biases]

    ew = []
    for i, p in enumerate(params):
        if i==len(params)-1 and gs>model.nclass:
            gs = model.nclass
        # quantize this weight
        q, s, err = quantize_q80(p, gs)
        # save the int8 weights to file
        serialize_int8(f, q) # save the tensor in int8
        serialize_fp32(f, s) # save scale factors
        # logging
        ew.append((err, p.shape))
        print(f"{i+1}/{len(params)} quantized {tuple(p.shape)} to Q8_0 with max error {err}")

    # print the highest error across all parameters, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")
    f.close()
    print(f"wrote {file_path}")
