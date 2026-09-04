#!/usr/bin/env python3
"""Export trained MLP weights into simple fixed-point .mem files."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_mlp import CLASSES, KwsMlp


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/checkpoints/kws_mlp.pt")
    parser.add_argument("--out", default="models/exported")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = KwsMlp()
    model.load_state_dict(checkpoint["model"])
    state = model.state_dict()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quant_params = {"classes": CLASSES, "layers": []}
    layer_map = [
        ("l1", "net.0.weight", "net.0.bias"),
        ("l2", "net.2.weight", "net.2.bias"),
        ("l3", "net.4.weight", "net.4.bias"),
    ]
    for name, weight_key, bias_key in layer_map:
        weights = state[weight_key].detach().cpu().numpy()
        biases = state[bias_key].detach().cpu().numpy()
        q_weights, weight_scale = quantize_symmetric(weights)
        q_biases = np.rint(biases / weight_scale).astype(np.int32)
        write_mem(out / f"weights_{name}.mem", q_weights, 8)
        write_mem(out / f"bias_{name}.mem", q_biases, 32)
        quant_params["layers"].append({
            "name": name,
            "weight_scale": weight_scale,
            "bias_scale": weight_scale,
            "shape": list(weights.shape),
        })

    with (out / "quant_params.json").open("w", encoding="utf-8") as f:
        json.dump(quant_params, f, indent=2)
    print(f"Exported fixed-point artifacts to {out}")


if __name__ == "__main__":
    main()
