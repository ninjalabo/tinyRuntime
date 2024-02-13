"""
Run simply with
$ pytest
"""

import os
import subprocess
import numpy as np
import re

def test_runc():
    """ Test run.c file """

    command = ["./run", "model.bin"]
    with open('err.txt', mode='wb') as fe:
        with open('stdout.txt', mode='wb') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    
    with open("stdout.txt", 'rb') as file:
        stdout_string = file.read().decode('utf-8')

    # define a regular expression pattern to match the accuracy
    pattern = r'Accuracy: (\d+\.\d+) %'

    # use re.search to find the match in the string
    match = re.search(pattern, stdout_string)
    assert match, "Couldn't read pattern in stdout.txt, string is now:{}".format(stdout_string)
    accuracy = float(match.group(1))
    assert accuracy > 95, "Too low accuracy < 95"

def test_runq():
    """ Test run.c file """

    command = ["./runq", "modelq8.bin"]
    with open('errq.txt', mode='wb') as fe:
        with open('stdoutq.txt', mode='wb') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)
            proc.wait()
    
    with open("stdoutq.txt", 'rb') as file:
        stdout_string = file.read().decode('utf-8')

    # define a regular expression pattern to match the accuracy
    pattern = r'Accuracy: (\d+\.\d+) %'

    # use re.search to find the match in the string
    match = re.search(pattern, stdout_string)
    assert match, "Couldn't read pattern in stdout.txt"
    accuracy = float(match.group(1))
    assert accuracy > 95, "Too low accuracy < 95"

