"""
Run simply with
$ pytest
"""

import os
import subprocess
import numpy as np


def test_runc():
    """ Test run.c file """

    command = ["./run", "model.bin"]
    with open('err.txt', mode='wb') as fe:
        with open('stdout.txt', mode='wb') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    
    inputs = [0.001, 1.57, 3.14, 6.28]
    stdout = np.fromfile("stdout.txt", sep="\t")
    assert np.allclose(stdout, np.sin([0.001, 1.57, 3.14, 6.28]), atol=0.1)


def test_runq():
    """ Test runq.c file """

    command = ["./runq", "modelq8.bin"]
    with open('errq.txt', mode='wb') as fe:
        with open('stdoutq.txt', mode='wb') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    
    inputs = [0.001, 1.57, 3.14, 6.28]
    stdout = np.fromfile("stdout.txt", sep="\t")
    print(np.abs(stdout - np.sin([0.001, 1.57, 3.14, 6.28])))
    assert np.allclose(stdout, np.sin([0.001, 1.57, 3.14, 6.28]), atol=0.1)

