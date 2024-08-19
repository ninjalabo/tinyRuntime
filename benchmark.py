# Don't edit this file! This was automatically generated from "benchmark.ipynb".

import subprocess
import os
import time
import psutil
import struct
import numpy as np
import pandas as pd
import torch
import csv
from datetime import datetime
import platform
from fastai.vision.all import *

from export import export_model, Quantizer, export_model_sq8
from train import load

def run_c(dir_path, base_command, model_path):
    '''Run C inference and return dictionary with accuracy, duration, model size, and memory usage.'''
    # get file paths of images and their labels
    files = []
    for label in range(10):
        sd_path = os.path.join(dir_path, str(label))
        f_paths = [os.path.join(sd_path, file) for file in os.listdir(sd_path)]
        files += f_paths

    # run C inference
    command = [base_command, model_path, *files]
    mems = []

    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc = psutil.Process(process.pid)
    while process.poll() == None:
        try:
            # `proc.memory_info().rss` returns the physical memory the process has used
            mems.append(proc.memory_info().rss / (1024 * 1024)) # append in megabytes
            time.sleep(0.1)  # check memory usage every 0.1 second
        except psutil.NoSuchProcess: # handle the case where the process ends abruptly
            pass
    end_time = time.time()

    output, _ = process.communicate()
    acc = float(output.decode().strip())
    dur = end_time - start_time
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    d = {"Accuracy": acc, "Time": dur, "Model size": model_size, "Memory usage": mems}

    return d


def run_python(dir_path, model_path):
    '''Run Python inference and return dictionary with accuracy, duration, model size, and memory usage.'''
    command = ["python", "run.py", model_path, dir_path]
    mems = []

    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc = psutil.Process(process.pid)
    while process.poll() == None:
        try:
            # `proc.memory_info().rss` returns the physical memory the process has used
            mems.append(proc.memory_info().rss / (1024 * 1024)) # append in megabytes
            time.sleep(0.1)  # check memory usage every 0.1 second
        except psutil.NoSuchProcess: # handle the case where the process ends abruptly
            pass
    end_time = time.time()

    output, _ = process.communicate()
    acc = float(output.decode().strip())
    dur = end_time - start_time
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    d = {"Accuracy": acc, "Time": dur, "Model size": model_size, "Memory usage": mems}

    return d

# Generate model files for tinyRuntime
path = "data"
model = load("resnet18").model
export_model(model, "model.bin")

dls = ImageDataLoaders.from_folder(untar_data(URLs.IMAGENETTE_320), valid='val', item_tfms=Resize(224),
                                   batch_tfms=Normalize.from_stats(*imagenet_stats), bs=64)
model_prepared, qmodel = Quantizer().quantize_one_batch(model, dls.one_batch()[0])
export_model_sq8(qmodel, model_prepared, "model-q8.bin")

def compare_results(res, architecture, runtime, quantized):
    '''Compare the results and fail if performance is worse compared to the previous result'''
    df = pd.read_csv("benchmark.csv")
    df = df[(df["Architecture"] == architecture) & (df["Runtime"] == runtime) & (df["Quantization"] == quantized)]
    if res["Accuracy"] < 0.9 * df["Accuracy"].values[-1]:
        raise ValueError(f"{runtime} - {quantized}: Accuracy is worse than 10%. Before: {df['Accuracy'].values[-1]}, Now: {res['Accuracy']}")
    if res["Time"] > 1.25 * df["Time"].values[-1]:
        raise ValueError(f"{runtime} - {quantized}: Time is worse than 25%. Before: {df['Time'].values[-1]}, Now: {res['Time']}")

def save_benchmark_csv():
    # Get results
    commit_id = os.getenv('GITHUB_SHA')
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    architecture = platform.machine()
    res0 = run_python(path, "model.pkl")
    res1 = run_c(path, "./run", "model.bin")
    res2 = run_c(path, "./runq", "model-q8.bin")
    # raise error if performance is worse than earlier
    compare_results(res0, architecture, "PyTorch", False)
    compare_results(res1, architecture, "tinyRuntime", False)
    compare_results(res2, architecture, "tinyRuntime", True)

    def generate_dict(res, runtime, quantization=False):
        d = {"Commit": commit_id, "Datetime": time, "Architecture": architecture, "Runtime": runtime,
             "Quantization": quantization, "Accuracy": res["Accuracy"], "Time": res["Time"],
             "Model size": res["Model size"], "Max memory": np.max(res["Memory usage"])}
        return d

    data = [generate_dict(res0, "PyTorch"), generate_dict(res1, "tinyRuntime"),
            generate_dict(res2, "tinyRuntime", quantization=True)]

    # Write results
    csv_file = "benchmark.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f'Data has been written to {csv_file}.')

save_benchmark_csv()
