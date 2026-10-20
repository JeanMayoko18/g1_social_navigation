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
    torch.manual_seed(0)
    np.random.seed(0)

    model = torch.jit.load(str(PT_PATH), map_location="cpu")
    model.eval()

    dummy = torch.randn(1, OBS_DIM, dtype=torch.float32)

    with torch.no_grad():
        torch_ref = model(dummy).detach().cpu().numpy()

    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=False,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes=None,
        #example_outputs=torch.from_numpy(torch_ref),
    )

    print(f"[OK] Exported: {ONNX_PATH}")

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model structure valid")

    sess = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    max_diffs = []

    for _ in range(20):
        x = torch.randn(1, OBS_DIM, dtype=torch.float32)

        with torch.no_grad():
            torch_out = model(x).detach().cpu().numpy()

        onnx_out = sess.run(
            ["actions"],
            {"obs": x.numpy()}
        )[0]

        diff = np.max(np.abs(torch_out - onnx_out))
        max_diffs.append(diff)

    print("Max diff over 20 tests:", max(max_diffs))
    print("Mean max diff:", float(np.mean(max_diffs)))

    if max(max_diffs) < 1e-4:
        print("[OK] ONNX conversion reliable")
    else:
        print("[WARN] ONNX conversion still not reliable")


if __name__ == "__main__":
    main()