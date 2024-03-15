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

from export import export_modelq8

file_path = "test/data/imagenette2/val_transformed/0/113"

# calculate reference values using Python model
model = torch.load("model.pt")
with open(file_path, "rb") as f:
    sizeof_float, nch, h, w = 4, 3, 224, 224
    image = torch.tensor(struct.unpack("f"*(nch*h*w), f.read(sizeof_float*nch*h*w))).view(1,nch,h,w)
    ref = model(image).detach()
    ref = torch.nn.functional.softmax(ref, dim=1).view(-1).numpy() # python model output

def execute(command):
    d = "test_outputs"
    os.makedirs(d, exist_ok=True)
    err = os.path.join(d, "err.txt")
    out = os.path.join(d, "stdout.txt")
    with open(err, mode="w") as fe:
        with open(out, mode="w") as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    res = np.loadtxt(out)

    return res

def test_runfiles(quantized=True, file_path=file_path):
    """ test run.c and runq.c works with an acceptable tolerance """
    # run vanilla model in test mode
    command = ["./run", "model.bin", file_path]
    res = execute(command)
    assert np.allclose(res, ref, atol=1e-5, rtol=0), "run.c: Probabilities are not close."

    if quantized:
        # run quantized model test with group size = 1 and 2 in test mode
        export_modelq8(file_path="modelq8_1.bin", gs=1)
        resq1 = execute(["./runq", "modelq8_1.bin", file_path])
        assert np.allclose(resq1, ref, atol=1e-5, rtol=0), "runq.c (group size = 1): Probabilities are not close."

        export_modelq8(file_path="modelq8_2.bin", gs=2)
        resq2 = execute(["./runq", "modelq8_2.bin", file_path])
        assert np.allclose(resq2, ref, atol=1e-2, rtol=0), "runq.c (group size = 2):Probabilities are not close."

    print("Done")
