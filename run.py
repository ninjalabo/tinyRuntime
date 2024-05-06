import os
import torch
import numpy as np
import struct
import sys

def run_python(model_path, dir_path):
    # get file paths of images and their labels
    files = []
    label_count = np.zeros(10, dtype=int)
    for label in range(10):
        sd_path = os.path.join(dir_path, str(label))
        f_paths = [os.path.join(sd_path, file) for file in os.listdir(sd_path)]
        label_count[label] = len(f_paths)
        files += f_paths

    # run Python inference
    bs = len(files)
    sizeof_float, nch, h, w = 4, 3, 224, 224
    probs = np.empty([len(files), 10], float)
    files = [files[i:i+bs] for i in range(0, len(files), bs)] # split files into batches

    model = torch.load(model_path)
    for i, batch in enumerate(files):
        bs = len(batch)
        imgs = torch.empty([bs, nch, h, w])
        for j, file in enumerate(batch):
            f = open(file, "rb")
            imgs[j] = torch.tensor(struct.unpack("f"*(nch*h*w), f.read())).view(1,nch,h,w)
            f.close()
        probs[i*bs:(i+1)*bs] = model(imgs).detach().view(bs, -1).numpy()

    labels = np.array([label for label in range(10) for _ in range(label_count[label])])
    preds = np.argmax(np.array(probs), axis=1)

    accuracy = 100 * np.mean(preds==labels)

    return accuracy

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <model> <data directory>")
        return
    
    model_path = sys.argv[1]
    dir_path = sys.argv[2]

    result = run_python(model_path, dir_path)
    print(result)

if __name__ == "__main__":
    main()
