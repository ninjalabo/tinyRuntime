# Don't edit this file! This was automatically generated from "prep.ipynb".

import os
import torchvision

torchvision.datasets.MNIST(root='./data', train=False, download=True)

base_path = './data/MNIST/raw/'
os.listdir(base_path)

path = os.path.join(base_path, 't10k-labels-idx1-ubyte')
print(os.stat(path).st_size)
with open(path, 'rb') as f:
    labels = list(f.read())

len(labels), labels[:12]

path = os.path.join('./data/MNIST', 'sorted')
os.makedirs(path, exist_ok=True)
for i in range(10):
    p = os.path.join(path, str(i))
    os.makedirs(p, exist_ok=True)
sorted(os.listdir(path))

labels = labels[8:]
labels = [(i, l) for i, l in enumerate(labels)]
labels[:9]

with open('./data/MNIST/raw/t10k-images-idx3-ubyte', 'rb') as f1:
    for i, l in labels:
        path = os.path.join('./data/MNIST/sorted', str(l), str(i))
        with open(path, 'wb') as f2:
            f1.seek(16+28*28*i, 0)
            x = f1.read(28*28)
            #show_image(x)
            f2.write(x)
            #break
