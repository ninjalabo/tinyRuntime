import torch

from export import export_model

model = torch.load("md/model.pkl", map_location="cpu")
export_model(model, "md/model.bin")
