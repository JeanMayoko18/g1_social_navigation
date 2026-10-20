#!/usr/bin/env python3

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path


PT_PATH = Path("policy_29dof.pt")
ONNX_PATH = Path("policy_29dof.onnx")

OBS_DIM = 96


def main():
    print(f"[INFO] Loading TorchScript model: {PT_PATH}")

    model = torch.jit.load(str(PT_PATH), map_location="cpu")
    model.eval()

    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)

    print(f"[INFO] Exporting ONNX with input shape: {tuple(dummy_obs.shape)}")

    torch.onnx.export(
        model,
        dummy_obs,
        str(ONNX_PATH),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "actions": {0: "batch_size"},
        },
    )

    print(f"[OK] Exported: {ONNX_PATH}")

    print("[INFO] Checking ONNX model...")
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX structure valid")

    print("[INFO] Comparing TorchScript vs ONNX Runtime...")
    x = torch.randn(1, OBS_DIM, dtype=torch.float32)

    with torch.no_grad():
        torch_out = model(x).detach().cpu().numpy()

    sess = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    onnx_out = sess.run(
        ["actions"],
        {"obs": x.cpu().numpy()}
    )[0]

    max_diff = np.max(np.abs(torch_out - onnx_out))

    print("Torch output shape:", torch_out.shape)
    print("ONNX output shape :", onnx_out.shape)
    print("Max abs diff      :", max_diff)

    if max_diff < 1e-4:
        print("[OK] Conversion reliable")
    else:
        print("[WARN] Difference higher than expected")


if __name__ == "__main__":
    main()