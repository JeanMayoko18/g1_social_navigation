#!/usr/bin/env python3

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path


PT_PATH = Path("policy_29dof.pt")
ONNX_PATH = Path("policy_29dof_lstm.onnx")

OBS_DIM = 96
HIDDEN_DIM = 256
ACTION_DIM = 29
SEQ_TEST_LEN = 100
EPS = 1e-8
EPS = 0.01


class RebuiltPolicy(nn.Module):
    """
    Rebuild equivalent of TorchScript policy:

        normalizer -> LSTM -> actor MLP

    Explicit state interface:

        obs, h0, c0 -> actions, h1, c1
    """

    def __init__(self):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=OBS_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=1
        )

        self.actor = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_DIM),
        )

        self.register_buffer("mean", torch.zeros(1, OBS_DIM))
        self.register_buffer("std", torch.ones(1, OBS_DIM))

    def forward(self, obs, h0, c0):
        # obs: [1, 96]
        x = (obs - self.mean) / (self.std + EPS)

        # LSTM input: [seq_len, batch, input_size]
        x = x.unsqueeze(0)

        y, (h1, c1) = self.rnn(x, (h0, c0))

        # y: [1, 1, 256] -> [1, 256]
        y = y.squeeze(0)

        actions = self.actor(y)

        return actions, h1, c1


def load_weights_from_torchscript(rebuilt, scripted):
    sd = scripted.state_dict()

    with torch.no_grad():
        # Normalizer
        rebuilt.mean.copy_(sd["normalizer._mean"])
        rebuilt.std.copy_(sd["normalizer._std"])

        # LSTM
        rebuilt.rnn.weight_ih_l0.copy_(sd["rnn.weight_ih_l0"])
        rebuilt.rnn.weight_hh_l0.copy_(sd["rnn.weight_hh_l0"])
        rebuilt.rnn.bias_ih_l0.copy_(sd["rnn.bias_ih_l0"])
        rebuilt.rnn.bias_hh_l0.copy_(sd["rnn.bias_hh_l0"])

        # Actor
        rebuilt.actor[0].weight.copy_(sd["actor.0.weight"])
        rebuilt.actor[0].bias.copy_(sd["actor.0.bias"])

        rebuilt.actor[2].weight.copy_(sd["actor.2.weight"])
        rebuilt.actor[2].bias.copy_(sd["actor.2.bias"])

        rebuilt.actor[4].weight.copy_(sd["actor.4.weight"])
        rebuilt.actor[4].bias.copy_(sd["actor.4.bias"])

        rebuilt.actor[6].weight.copy_(sd["actor.6.weight"])
        rebuilt.actor[6].bias.copy_(sd["actor.6.bias"])


def reset_torchscript_state(scripted):
    with torch.no_grad():
        scripted.hidden_state.zero_()
        scripted.cell_state.zero_()


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"[INFO] Loading TorchScript: {PT_PATH}")
    scripted = torch.jit.load(str(PT_PATH), map_location="cpu")
    scripted.eval()

    rebuilt = RebuiltPolicy()
    rebuilt.eval()

    print("[INFO] Copying weights...")
    load_weights_from_torchscript(rebuilt, scripted)

    # ------------------------------------------------------------
    # First: validate rebuilt PyTorch vs original TorchScript
    # ------------------------------------------------------------
    print("[INFO] Validating rebuilt PyTorch model vs TorchScript...")

    reset_torchscript_state(scripted)

    h = torch.zeros_like(scripted.hidden_state)
    c = torch.zeros_like(scripted.cell_state)

    rebuild_diffs = []

    for t in range(SEQ_TEST_LEN):
        obs = torch.randn(1, OBS_DIM)

        with torch.no_grad():
            y_ts = scripted(obs)
            y_rb, h, c = rebuilt(obs, h, c)

        diff = torch.max(torch.abs(y_ts - y_rb)).item()
        rebuild_diffs.append(diff)

    print("Rebuilt max diff :", max(rebuild_diffs))
    print("Rebuilt mean diff:", float(np.mean(rebuild_diffs)))

    if max(rebuild_diffs) >= 1e-5:
        print("[WARN] Rebuilt model does not match TorchScript enough.")
        print("       Do not export/deploy before fixing this.")
        return

    print("[OK] Rebuilt model matches TorchScript")

    # ------------------------------------------------------------
    # Export ONNX
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Validate ONNX Runtime vs TorchScript over sequence
    # ------------------------------------------------------------
    print("[INFO] Validating ONNX Runtime vs TorchScript over sequence...")

    sess = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    reset_torchscript_state(scripted)

    h_np = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)
    c_np = np.zeros((1, 1, HIDDEN_DIM), dtype=np.float32)

    onnx_diffs = []

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
            }
        )

        diff = np.max(np.abs(y_ts - y_onnx))
        onnx_diffs.append(diff)

        if t < 10 or t == SEQ_TEST_LEN - 1:
            print(f"step={t:03d} diff={diff:.8f}")

    max_diff = float(np.max(onnx_diffs))
    mean_diff = float(np.mean(onnx_diffs))

    print("\n========== FINAL RESULT ==========")
    print("ONNX max diff :", max_diff)
    print("ONNX mean diff:", mean_diff)

    if max_diff < 1e-4:
        print("[OK] ONNX LSTM policy is reliable")
        print(f"[OK] Use this file: {ONNX_PATH}")
    else:
        print("[WARN] ONNX conversion is not reliable yet")


if __name__ == "__main__":
    main()