import torch
import os

from export import export_model, export_modelq8

model = torch.load(f"models/model.pkl")
print(model)
export_model(model, f"models/model.bin")
export_modelq8(model, f"models/modelq8.bin")
