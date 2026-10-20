#!/usr/bin/env python3

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path


PT_PATH = Path("motion.pt")
ONNX_PATH = Path("motion_lstm.onnx")

OBS_DIM = 47
HIDDEN_DIM = 64
ACTION_DIM = 12
SEQ_TEST_LEN = 100
TOL_REBUILD = 1e-5
TOL_ONNX = 1e-4


class RebuiltPolicy(nn.Module):
    """
    Rebuild equivalent of the official Unitree TorchScript policy.

    TorchScript structure:

        memory: LSTM(input=47, hidden=64)
        actor:
            Linear(64 -> 32)
            ELU()
            Linear(32 -> 12)

    Explicit ONNX interface:

        obs, h0, c0 -> actions, h1, c1
    """

    def __init__(self):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=OBS_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=1,
        )

        self.actor = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.ELU(),
            nn.Linear(32, ACTION_DIM),
        )

    def forward(self, obs, h0, c0):
        # obs: [1, 47]
        # h0/c0: [1, 1, 64]

        x = obs.unsqueeze(0)  # [1, 1, 47]

        y, (h1, c1) = self.rnn(x, (h0, c0))

        y = y.squeeze(0)  # [1, 64]

        actions = self.actor(y)  # [1, 12]

        return actions, h1, c1


def load_weights_from_torchscript(rebuilt, scripted):
    """
    Copy weights from motion.pt TorchScript into the rebuilt PyTorch model.
    """

    sd = scripted.state_dict()

    required_keys = [
        "hidden_state",
        "cell_state",
        "memory.weight_ih_l0",
        "memory.weight_hh_l0",
        "memory.bias_ih_l0",
        "memory.bias_hh_l0",
        "actor.0.weight",
        "actor.0.bias",
        "actor.2.weight",
        "actor.2.bias",
    ]

    missing = [k for k in required_keys if k not in sd]

    if missing:
        raise KeyError(f"Missing keys in TorchScript state_dict: {missing}")

    with torch.no_grad():
        rebuilt.rnn.weight_ih_l0.copy_(sd["memory.weight_ih_l0"])
        rebuilt.rnn.weight_hh_l0.copy_(sd["memory.weight_hh_l0"])
        rebuilt.rnn.bias_ih_l0.copy_(sd["memory.bias_ih_l0"])
        rebuilt.rnn.bias_hh_l0.copy_(sd["memory.bias_hh_l0"])

        rebuilt.actor[0].weight.copy_(sd["actor.0.weight"])
        rebuilt.actor[0].bias.copy_(sd["actor.0.bias"])

        rebuilt.actor[2].weight.copy_(sd["actor.2.weight"])
        rebuilt.actor[2].bias.copy_(sd["actor.2.bias"])


def reset_torchscript_state(scripted):
    """
    Reset TorchScript internal recurrent state.
    """

    with torch.no_grad():
        scripted.hidden_state.zero_()
        scripted.cell_state.zero_()


def validate_rebuilt_vs_torchscript(scripted, rebuilt):
    """
    Validate rebuilt PyTorch model against original TorchScript over a sequence.
    """

    print("[INFO] Validating rebuilt PyTorch model vs TorchScript...")

    reset_torchscript_state(scripted)

    h = torch.zeros_like(scripted.hidden_state)
    c = torch.zeros_like(scripted.cell_state)

    diffs = []

    for t in range(SEQ_TEST_LEN):
        obs = torch.randn(1, OBS_DIM, dtype=torch.float32)

        with torch.no_grad():
            y_ts = scripted(obs)
            y_rb, h, c = rebuilt(obs, h, c)

        diff = torch.max(torch.abs(y_ts - y_rb)).item()
        diffs.append(diff)

        if t < 5 or t == SEQ_TEST_LEN - 1:
            print(f"[REBUILD] step={t:03d} diff={diff:.8f}")

    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))

    print("[REBUILD] max diff :", max_diff)
    print("[REBUILD] mean diff:", mean_diff)

    if max_diff >= TOL_REBUILD:
        raise RuntimeError(
            f"Rebuilt model mismatch: max_diff={max_diff} >= {TOL_REBUILD}"
        )

    print("[OK] Rebuilt model matches TorchScript")


def export_onnx(rebuilt):
    """
    Export rebuilt recurrent policy to ONNX.
    """

    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    dummy_h = torch.zeros(1, 1, HIDDEN_DIM, dtype=torch.float32)
    dummy_c = torch.zeros(1, 1, HIDDEN_DIM, dtype=torch.float32)

    print(f"[INFO] Exporting ONNX: {ONNX_PATH}")

    torch.onnx.export(
        rebuilt,
        (dummy_obs, dummy_h, dummy_c),
        str(ONNX_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["obs", "h0", "c0"],
        output_names=["actions", "h1", "c1"],
        dynamic_axes=None,
    )

    print("[OK] Exported ONNX")

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    print("[OK] ONNX structure valid")


def validate_onnx_vs_torchscript(scripted):
    """
    Validate ONNX Runtime output against original TorchScript.
    """

    print("[INFO] Validating ONNX Runtime vs TorchScript...")

    sess = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    reset_torchscript_state(scripted)

    h_np = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)
    c_np = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)

    diffs = []

    for t in range(SEQ_TEST_LEN):
        obs = torch.randn(1, OBS_DIM, dtype=torch.float32)

        with torch.no_grad():
            y_ts = scripted(obs).cpu().numpy()

        y_onnx, h_np, c_np = sess.run(
            ["actions", "h1", "c1"],
            {
                "obs": obs.numpy(),
                "h0": h_np,
                "c0": c_np,
            },
        )

        diff = float(np.max(np.abs(y_ts - y_onnx)))
        diffs.append(diff)

        if t < 10 or t == SEQ_TEST_LEN - 1:
            print(f"[ONNX] step={t:03d} diff={diff:.8f}")

    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))

    print("\n========== FINAL RESULT ==========")
    print("ONNX max diff :", max_diff)
    print("ONNX mean diff:", mean_diff)

    if max_diff < TOL_ONNX:
        print("[OK] ONNX LSTM policy is reliable")
        print(f"[OK] Use this file: {ONNX_PATH}")
    else:
        raise RuntimeError(
            f"ONNX mismatch: max_diff={max_diff} >= {TOL_ONNX}"
        )


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    if not PT_PATH.exists():
        raise FileNotFoundError(f"Missing TorchScript file: {PT_PATH}")

    print(f"[INFO] Loading TorchScript: {PT_PATH}")

    scripted = torch.jit.load(
        str(PT_PATH),
        map_location="cpu",
    )

    scripted.eval()

    print("[INFO] TorchScript state_dict keys:")
    for k, v in scripted.state_dict().items():
        print(f"  {k}: {tuple(v.shape)}")

    rebuilt = RebuiltPolicy()
    rebuilt.eval()

    print("[INFO] Copying weights...")
    load_weights_from_torchscript(rebuilt, scripted)

    validate_rebuilt_vs_torchscript(scripted, rebuilt)

    export_onnx(rebuilt)

    validate_onnx_vs_torchscript(scripted)


if __name__ == "__main__":
    main()