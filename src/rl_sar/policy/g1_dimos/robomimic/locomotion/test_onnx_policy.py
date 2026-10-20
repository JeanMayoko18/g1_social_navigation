#!/usr/bin/env python3

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path


PT_PATH = Path("policy_29dof.pt")
ONNX_PATH = Path("policy_29dof.onnx")

OBS_DIM = 96
HISTORY_LEN = 1
INPUT_DIM = OBS_DIM * HISTORY_LEN


def main():
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model structure valid")

    torch_model = torch.jit.load(str(PT_PATH), map_location="cpu")
    torch_model.eval()

    x = torch.randn(1, INPUT_DIM, dtype=torch.float32)

    with torch.no_grad():
        torch_out = torch_model(x).cpu().numpy()

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["actions"], {"obs": x.cpu().numpy()})[0]

    max_diff = np.max(np.abs(torch_out - onnx_out))

    print("Torch output shape:", torch_out.shape)
    print("ONNX output shape :", onnx_out.shape)
    print("Max abs diff      :", max_diff)

    if max_diff < 1e-4:
        print("[OK] ONNX output matches Torch")
    else:
        print("[WARN] Difference is higher than expected")


if __name__ == "__main__":
    main()