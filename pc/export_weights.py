#!/usr/bin/env python3
"""Export the trained MLP as a faithful fixed-point (integer) reference.

This module is the single source of truth for the Python-to-VHDL numeric
contract. It quantizes the model to int8 weights / int32 biases, calibrates a
per-layer requantization right-shift, and runs an *integer-only* forward pass
that the VHDL datapath must reproduce bit-for-bit.

Outputs (under --out, default models/exported):
  weights_l{1,2,3}.mem   int8 weights, hex, row-major [out][in]
  bias_l{1,2,3}.mem       int32 biases, hex
  quant_params.json       scales, shifts, shapes, input_scale, accuracies
  golden_vectors.csv      expected_class, fp_predicted_class, float_predicted_class, 980 int8 features

Simulation-friendly copies (under fpga/sim/vectors):
  golden_features.txt     decimal int8, one value per line, rows concatenated
  golden_labels.txt       one fp_predicted_class per line (the VHDL target)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from capture_mfcc import FEATURE_DIM
from train_mlp import CLASSES, KwsMlp


LAYER_MAP = [
    ("l1", "net.0.weight", "net.0.bias"),
    ("l2", "net.3.weight", "net.3.bias"),
    ("l3", "net.6.weight", "net.6.bias"),
]


def quantize_symmetric(arr: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(arr)))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = np.clip(np.rint(arr / scale), -128, 127).astype(np.int8)
    return q, scale


def write_mem(path: Path, values: np.ndarray, width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = values.reshape(-1)
    mask = (1 << width) - 1
    digits = width // 4
    with path.open("w", encoding="utf-8") as f:
        for value in flat:
            f.write(f"{int(value) & mask:0{digits}X}\n")


def requant(acc: np.ndarray, shift: int, relu: bool) -> np.ndarray:
    """Integer requantization identical to the VHDL: optional ReLU, then a
    round-half-up arithmetic right shift, then clamp to int8 [0, 127] (hidden)."""
    v = np.maximum(acc, 0) if relu else acc
    if shift > 0:
        v = (v + (1 << (shift - 1))) >> shift
    return np.clip(v, 0, 127).astype(np.int32)


def calibrate_shift(acc: np.ndarray) -> int:
    """Choose the smallest shift that keeps post-ReLU activations within int8."""
    peak = int(np.max(np.maximum(acc, 0))) if acc.size else 0
    shift = 0
    while (peak >> shift) > 127:
        shift += 1
    return shift


class FixedPointModel:
    """Integer-only MLP that the VHDL must match exactly."""

    def __init__(self, qw: list[np.ndarray], qb: list[np.ndarray], shifts: list[int]) -> None:
        self.qw = qw          # int32 arrays, shape [out, in]
        self.qb = qb          # int32 arrays, shape [out]
        self.shifts = shifts  # per-layer right shift (last layer unused)

    def forward(self, x_q: np.ndarray) -> np.ndarray:
        """x_q: [N, 980] int; returns argmax class per row."""
        a = x_q.astype(np.int32)
        # Hidden layers: acc = W.x + b (int32), ReLU + requant shift -> int8.
        for li in range(2):
            acc = a @ self.qw[li].T + self.qb[li]        # [N, out]
            a = requant(acc, self.shifts[li], relu=True)
        logits = a @ self.qw[2].T + self.qb[2]           # [N, 12] int32
        return logits.argmax(axis=1)


def build_fixed_point(state: dict, input_scale: float, calib_x_q: np.ndarray):
    """Quantize weights, compute int32 biases with the correct cumulative input
    scale, and calibrate per-layer shifts using integer forward passes."""
    qw: list[np.ndarray] = []
    qb: list[np.ndarray] = []
    scales: list[float] = []
    shifts: list[int] = []

    s_in = input_scale
    a = calib_x_q.astype(np.int32)
    for idx, (_, wkey, bkey) in enumerate(LAYER_MAP):
        w = state[wkey].detach().cpu().numpy()
        b = state[bkey].detach().cpu().numpy()
        w_q, s_w = quantize_symmetric(w)
        # Bias shares the accumulator scale s_in * s_w.
        b_q = np.rint(b / (s_in * s_w)).astype(np.int32)
        qw.append(w_q.astype(np.int32))
        qb.append(b_q)
        scales.append(s_w)

        acc = a @ w_q.astype(np.int32).T + b_q
        if idx < 2:
            shift = calibrate_shift(acc)
            shifts.append(shift)
            a = requant(acc, shift, relu=True)
            s_in = s_in * s_w * (2 ** shift)   # scale of this layer's int8 output
        else:
            shifts.append(0)                    # final layer: argmax, no requant
    return FixedPointModel(qw, qb, shifts), scales, shifts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/checkpoints/kws_mlp.pt")
    parser.add_argument("--out", default="models/exported")
    parser.add_argument("--cache", help="MFCC cache used for calibration and golden vectors")
    parser.add_argument("--sim-out", default="fpga/sim/vectors", help="Directory for VHDL sim vectors")
    parser.add_argument("--vectors", type=int, default=128)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = KwsMlp()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    state = model.state_dict()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.cache:
        raise SystemExit("--cache is required to calibrate the fixed-point model and emit golden vectors")

    cache = np.load(args.cache, allow_pickle=False)
    x = cache["x"].astype(np.float32)
    y = cache["y"].astype(np.int64)
    split = cache["split"]
    val_idx = np.where(split == "val")[0]
    if val_idx.size == 0:
        raise SystemExit(f"No validation rows in cache: {args.cache}")

    # Normalize with the training statistics, then quantize the input to int8.
    x_val_norm = ((x[val_idx] - mean) / std).astype(np.float32)
    _, input_scale = quantize_symmetric(x_val_norm)
    x_val_q = np.clip(np.rint(x_val_norm / input_scale), -128, 127).astype(np.int32)

    fp_model, scales, shifts = build_fixed_point(state, input_scale, x_val_q)

    # Accuracy report: float vs fixed-point, both against the true labels.
    with torch.no_grad():
        float_pred_val = model(torch.from_numpy(x_val_norm)).argmax(dim=1).numpy()
    fp_pred_val = fp_model.forward(x_val_q)
    float_acc = float(np.mean(float_pred_val == y[val_idx]))
    fp_acc = float(np.mean(fp_pred_val == y[val_idx]))
    agree = float(np.mean(fp_pred_val == float_pred_val))
    print(f"val float_acc={float_acc:.4f}  fixed_acc={fp_acc:.4f}  fp/float agreement={agree:.4f}")
    print(f"input_scale={input_scale:.6g}  shifts={shifts}")

    # Emit weights and biases.
    quant_params = {
        "classes": CLASSES,
        "feature_dim": FEATURE_DIM,
        "input_scale": input_scale,
        "float_val_acc": float_acc,
        "fixed_val_acc": fp_acc,
        "layers": [],
    }
    for (name, _, _), w_q, b_q, s_w, shift in zip(LAYER_MAP, fp_model.qw, fp_model.qb, scales, shifts):
        write_mem(out / f"weights_{name}.mem", w_q, 8)
        write_mem(out / f"bias_{name}.mem", b_q, 32)
        quant_params["layers"].append({
            "name": name,
            "weight_scale": s_w,
            "shift": int(shift),
            "shape": list(w_q.shape),
        })
    with (out / "quant_params.json").open("w", encoding="utf-8") as f:
        json.dump(quant_params, f, indent=2)

    # Golden vectors (human/contract CSV + simulation-friendly text files).
    limit = min(args.vectors, val_idx.size)
    sel_q = x_val_q[:limit]
    sel_expected = y[val_idx][:limit]
    sel_float = float_pred_val[:limit]
    sel_fp = fp_pred_val[:limit]

    with (out / "golden_vectors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["expected_class", "fp_predicted_class", "float_predicted_class"]
            + [f"x{i}" for i in range(FEATURE_DIM)]
        )
        for exp, fp, fl, feats in zip(sel_expected, sel_fp, sel_float, sel_q):
            writer.writerow([int(exp), int(fp), int(fl)] + [int(v) for v in feats])

    sim_dir = Path(args.sim_out)
    sim_dir.mkdir(parents=True, exist_ok=True)
    with (sim_dir / "golden_features.txt").open("w", encoding="utf-8") as f:
        for feats in sel_q:
            for v in feats:
                f.write(f"{int(v)}\n")
    with (sim_dir / "golden_labels.txt").open("w", encoding="utf-8") as f:
        f.write(f"{limit}\n")
        for fp in sel_fp:
            f.write(f"{int(fp)}\n")

    print(f"Exported fixed-point artifacts to {out} and sim vectors ({limit} rows) to {sim_dir}")


if __name__ == "__main__":
    main()
