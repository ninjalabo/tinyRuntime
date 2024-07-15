# Don't edit this file! This was automatically generated from "benchmark.ipynb".

import subprocess
import os
import time
import psutil
import struct
import numpy as np
import torch
import csv
from datetime import datetime
import platform

from export import export_model, export_modelq8
from train import load

def run_c(dir_path, base_command, model_path):
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

    return acc, dur, model_size, mems


def run_python(dir_path, model_path):
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

    return acc, dur, model_size, mems

# Generate model files for tinyRuntime
path = "data"
model = load("resnet18").model
export_model(model, "model.bin")
export_modelq8(model, "model-q8.bin")

def save_benchmark_csv():
    # Get results
    commit_id = os.getenv('GITHUB_SHA')
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    architecture = platform.machine()
    res0 = run_python(path, "model.pkl")
    res1 = run_c(path, "./run", "model.bin")
    res2 = run_c(path, "./runq", "model-q8.bin")

    def generate_dict(res, runtime, quantization=False):
        d = {"Commit": commit_id, "Datetime": time, "Architecture": architecture, "Runtime": runtime,
             "Quantization": quantization, "Accuracy": res[0], "Time": res[1], "Model size": res[2],
             "Max memory": np.max(res[3])}
        return d

    data = [generate_dict(res0, "PyTorch"), generate_dict(res1, "tinyRuntime"),
            generate_dict(res2, "tinyRuntime", quantization=True)]

    # Write results
    csv_file = "benchmark.csv"
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        # If the file is empty, write header
        if f.tell() == 0:
            writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f'Data has been appended to {csv_file}.')

save_benchmark_csv()
