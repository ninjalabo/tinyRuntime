import pytest
import subprocess
from train import load
from export import export_model, export_modelq8

model = load("resnet18").model
export_model(model, "resnet18.bin")
export_modelq8(model, "resnet18-q8.bin")

def run_vanilla():
    command = ["./run", "resnet18.bin", "data/*/*"]
    return subprocess.run(command)

def run_quantized():
    command = ["./runq", "resnet18-q8.bin", "data/*/*"]
    return subprocess.run(command)

def test_vanilla_tinyruntime(benchmark):
    benchmark(run_vanilla)

def test_quantized_tinyruntime(benchmark):
    benchmark(run_quantized)
