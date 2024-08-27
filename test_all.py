# Don't edit this file! This was automatically generated from "test_all.ipynb".

"""
Run simply with
$ pytest
"""
import pytest

import os
import subprocess
import numpy as np
import torch
import struct
# import fastai classes/functions individually to avoid conflicts with pytest
from fastai.vision.all import ImageDataLoaders, untar_data, URLs, Resize, Normalize, imagenet_stats

from export import export_model, export_model_dq8, export_model_sq8, Quantizer
from train import load

file_path = "test/imagenette2-320/data/5/4.bin"

def calculate_reference_values(model, file_path):
    # calculate reference values using Python model
    with open(file_path, "rb") as f:
        sizeof_float, nch, h, w = 4, 3, 224, 224
        image = torch.tensor(struct.unpack("f"*(nch*h*w), f.read(sizeof_float*nch*h*w))).view(1,nch,h,w)
        ref = model(image).detach()
        ref = torch.nn.functional.softmax(ref, dim=1).view(-1).numpy() # python model output
    return ref

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

@pytest.mark.parametrize("model_size, blas",
                         [(18, "OFF"), (34, "OFF"), (50, "OFF"),
                         (18, "ON"), (34, "ON"), (50, "ON")])
def test_vanilla(model_size, blas, file_path=file_path):
    """ test run.c works with an acceptable tolerance """
    model = load(f"resnet{model_size}").model
    ref = calculate_reference_values(model, file_path)

    # TEST vanilla inference
    subprocess.run(["make", "compile", "QUANT_TYPE=DQ", f"BLAS={blas}"])
    export_model(model, file_path="test.bin")
    command = ["./run", "test", "test.bin", file_path]
    res = execute(command)
    assert np.allclose(res, ref, atol=1e-5, rtol=0), "run.c: Probabilities are not close."

@pytest.mark.parametrize("model_size, blas",
                         [(18, "OFF"), (34, "OFF"), (50, "OFF"),
                         (18, "ON"), (34, "ON"), (50, "ON")])
def test_dq(model_size, blas, file_path=file_path):
    """ test runq.c (QUANT_TYPE=DQ) works with an acceptable tolerance """
    model = load(f"resnet{model_size}").model
    ref = calculate_reference_values(model, file_path)
    # symmetric (no zero point) with group size = 1
    export_model_dq8(model, file_path="test.bin", gs=1, asymmetric=False)
    res = execute(["./runq", "test", "test.bin", file_path])
    assert np.allclose(res, ref, atol=1e-5, rtol=0), "runq.c (DQ, symmetric, gs=1): Probabilities are not close."
    # symmetric with group size = 2
    export_model_dq8(model, file_path="test.bin", gs=2, asymmetric=False)
    res = execute(["./runq", "test", "test.bin", file_path])
    assert np.allclose(res, ref, atol=2e-2, rtol=0), "runq.c (DQ, symmetric, gs=2): Probabilities are not close."
    # asymmetric (i.e. using zero point) with group size = 1
    export_model_dq8(model, file_path="test.bin", gs=10, asymmetric=True)
    res = execute(["./runq", "test", "test.bin", file_path])
    assert np.allclose(res, ref, atol=1e-1, rtol=0), "runq.c (DQ, asymmetric, gs=10): Probabilities are not close."

# Quantize and export models beforehand to avoid repetition of static quantization which is a heavy process
path = untar_data(URLs.IMAGENETTE_320)
dls = ImageDataLoaders.from_folder(path, valid='val', item_tfms=Resize(224),
                                   batch_tfms=Normalize.from_stats(*imagenet_stats, cuda=False),
                                   device=torch.device("cpu"))

def quantize_and_export(model_size):
    learn = load(f"resnet{model_size}")
    model_prepared, qmodel = Quantizer().quantize_one_batch(learn.model, dls.one_batch()[0])
    export_model_sq8(qmodel, model_prepared, file_path=f"test{model_size}.bin")
    ref = calculate_reference_values(qmodel, file_path)
    return ref

refs = []
for size in [18, 34, 50]:
    refs.append(quantize_and_export(size))

@pytest.mark.parametrize("model_size, blas, ref",
                         [(18, "OFF", refs[0]), (34, "OFF", refs[1]), (50, "OFF", refs[2]),
                         (18, "ON", refs[0]), (34, "ON", refs[1]), (50, "ON", refs[2])])
def test_sq(model_size, blas, ref, file_path=file_path):
    """ test runq.c (QUANT_TYPE=SQ) works with an acceptable tolerance """
    subprocess.run(["make", "compile", f"BLAS={blas}"])
    res = execute(["./runq", "test", f"test{model_size}.bin", file_path])
    # FIX: Tolerance is now large because quantized model inference on x86 is different.
    # Our current tinyRuntime follows quantized model inference on ARM (qnnpack)
    assert np.allclose(res, ref, atol=1e-1, rtol=0), "runq.c (SQ): Probabilities are not close."
