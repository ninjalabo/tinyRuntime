# Don't edit this file! This was automatically generated from "evaluate.ipynb".

import subprocess
import numpy as np
import os
import itertools
import sys

def calculate_accuracy(model, quantized=False):
    fe = open('err.txt', mode='wb')
    fo = open('stdout.txt', mode='wb')
    
    if quantized:
        base_command = ["./runq", model]
    else:
        base_command = ["./run", model]
    
    base_path = "data/MNIST/sorted"
    label_count = np.zeros(10, dtype=int)
    for label in range(10):
        dir_path = os.path.join(base_path, str(label))
        for root, _, files in os.walk(dir_path):
            command = base_command + [os.path.join(root, file) for file in files]
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
            label_count[label] = len(files)
    
    fe.close()
    fo.close()
    
    labels = np.array(list(itertools.chain(*[[i]*label_count[i] for i in range(10)])))
    assert len(labels)==10000 # MNIST test set has size 10000
    probs = np.loadtxt("stdout.txt")
    preds = np.argmax(probs, axis=1)
    
    return np.mean(preds==labels)
