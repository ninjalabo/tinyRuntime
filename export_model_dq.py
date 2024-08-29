import torch
import json

from export import export_model_dq8

model = torch.load("md/model.pkl", map_location="cpu")
with open("md/runtime_info.json", "r") as f:
    runtime_info = json.load(f)
asymmetric = runtime_info.get("compression", {}).get("quantization", {}).get("asymmetric")
export_model_dq8(model, "md/model.bin", gs=64, asymmetric=asymmetric)
