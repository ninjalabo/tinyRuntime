"""
Run simply with
$ pytest
"""

import os
import subprocess
import numpy as np
import torch
from torch import nn
import struct

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        dim = 128
        self.dim = dim
        self.nclass = 10
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList([nn.Linear(28*28, dim), 
                                    nn.Linear(dim, dim//2)])
        self.activation = nn.ReLU()
        self.out = nn.Linear(dim//2, self.nclass)
        
    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.out(x)
        return x

def read_model():
    ''' read model.bin and assign parameters to the model '''
    model = Model()
    f = open("model.bin", "rb")
    inp_size = 28*28
    dim = struct.unpack('i', f.read(4))[0]
    nclass = struct.unpack('i', f.read(4))[0]
    dim2 = dim//2
    wi = torch.tensor(struct.unpack('f'*(inp_size*dim), f.read(4*inp_size*dim))).view(dim,inp_size)
    wh = torch.tensor(struct.unpack('f'*(dim*dim2), f.read(4*dim*dim2))).view(dim2,dim)
    wo = torch.tensor(struct.unpack('f'*(dim2*nclass), f.read(4*dim2*nclass))).view(nclass, dim2)
    bi = torch.tensor(struct.unpack('f'*(dim), f.read(4*dim)))
    bh = torch.tensor(struct.unpack('f'*(dim2), f.read(4*dim2)))
    bo = torch.tensor(struct.unpack('f'*nclass, f.read(4*nclass)))
    f.close()

    i = 0
    params = [wi, bi, wh, bh, wo, bo]
    for p in model.parameters():
        p.data = params[i]
        i += 1

    return model

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

def export_modelq8(model=None, filepath="modelq8.bin", gs=64):
    ''' read a model from model.bin if not given and export a quatized (int8) model to filepath '''
    if model==None:
        model = read_model()

    f = open(filepath, "wb")
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
    print(f"wrote {filepath}")

def test_runfiles():
    """ test run.c and runq.c works with an acceptable tolerance """

    # run vanilla model in test mode
    command = ["./run", "model.bin", "-m", "test"]
    with open('err.txt', mode='wb') as fe:
        with open('stdout.txt', mode='wb') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    res = np.fromfile("stdout.txt", sep="\n")

    # run quantized model test with group size = 1 in test mode
    export_modelq8(filepath="modelq8_1.bin", gs=1)
    command = ["./runq", "modelq8_1.bin", "-m", "test"]
    with open('err1.txt', mode='wb') as fe:
        with open('stdout1.txt', mode='wb') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    res1 = np.fromfile("stdout1.txt", sep="\n")

    # run quantized model test with group size = 1 in test mode
    export_modelq8(filepath="modelq8_2.bin", gs=2)
    command = ["./runq", "modelq8_2.bin", "-m", "test"]
    with open('err2.txt', mode='wb') as fe:
        with open('stdout2.txt', mode='wb') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    res2 = np.fromfile("stdout2.txt", sep="\n")

    # retrieve reference values using Python model
    model = read_model()
    file_path = "data/MNIST/sorted/7/1012"
    with open(file_path, "rb") as f:
        image = torch.tensor(struct.unpack('B'*(28*28), f.read(28*28))).view(1,28,28)
        image = ((image/255 - 0.5) / 0.5)
        ref = model(image).detach()
        ref = torch.nn.functional.softmax(ref, dim=1).view(-1).numpy() # python model output

    assert np.allclose(res, ref, atol=1e-5, rtol=0), "run.c: Probabilities are not close."
    assert np.allclose(res1, ref, atol=1e-5, rtol=0), "runq.c (group size = 1): Probabilities are not close."
    assert np.allclose(res2, ref, atol=1e-2, rtol=0), "runq.c (group size = 2):Probabilities are not close."
