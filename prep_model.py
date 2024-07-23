import torch
import json

from export import export_model, export_modelq8

with open("md/runtime_info.json", "r") as f:
    runtime_info = json.load(f)
quantization = runtime_info.get("compression", {}).get("quantization", False)

model = torch.load("md/model.pkl", map_location="cpu")

export_function = export_modelq8 if quantization else export_model
export_function(model, "md/model.bin")
