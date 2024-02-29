# Don't edit this file! This was automatically generated from "test_all.ipynb".

"""
Run simply with
$ pytest
"""

import os
import subprocess
import numpy as np
import torch
import struct

from model import Model
from export import export_modelq8

def test_runfiles():
    """ test run.c and runq.c works with an acceptable tolerance """

    file_path = "test/data/MNIST/sorted/7/1012"
    d = "test_outputs"
    os.makedirs(d, exist_ok=True)
    # run vanilla model in test mode
    err = os.path.join(d, 'err.txt')
    stdout = os.path.join(d, 'stdout.txt')
    command = ["./run", "model.bin", file_path]
    with open(err, mode='w') as fe:
        with open(stdout, mode='w') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    res = np.loadtxt(stdout)

    # run quantized model test with group size = 1 in test mode
    err1 = os.path.join(d, 'err1.txt')
    stdout1 = os.path.join(d, 'stdout1.txt')
    export_modelq8(file_path="modelq8_1.bin", gs=1)
    command = ["./runq", "modelq8_1.bin", file_path]
    with open(err1, mode='w') as fe:
        with open(stdout1, mode='w') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    res1 = np.loadtxt(stdout1)

    # run quantized model test with group size = 1 in test mode
    err2 = os.path.join(d, 'err2.txt')
    stdout2 = os.path.join(d, 'stdout2.txt')
    export_modelq8(file_path="modelq8_2.bin", gs=2)
    command = ["./runq", "modelq8_2.bin", file_path]
    with open(err2, mode='w') as fe:
        with open(stdout2, mode='w') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    res2 = np.loadtxt(stdout2)

    # retrieve reference values using Python model
    model = torch.load("model.pt")
    with open(file_path, "rb") as f:
        image = torch.tensor(struct.unpack('B'*(28*28), f.read(28*28))).view(1,1,28,28)
        image = ((image/255 - 0.5) / 0.5)
        ref = model(image).detach()
        ref = torch.nn.functional.softmax(ref, dim=1).view(-1).numpy() # python model output

    assert np.allclose(res, ref, atol=1e-5, rtol=0), "run.c: Probabilities are not close."
    assert np.allclose(res1, ref, atol=1e-5, rtol=0), "runq.c (group size = 1): Probabilities are not close."
    assert np.allclose(res2, ref, atol=1e-2, rtol=0), "runq.c (group size = 2):Probabilities are not close."
