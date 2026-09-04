#!/usr/bin/env python3
"""Export trained MLP weights into simple fixed-point .mem files."""
import argparse
import csv
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


def write_vector_csv(path: Path, model: KwsMlp, checkpoint: dict, cache_path: Path, limit: int) -> None:
    cache = np.load(cache_path, allow_pickle=False)
    x = cache["x"].astype(np.float32)
    y = cache["y"].astype(np.int64)
    split = cache["split"]
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    mask = np.where(split == "val")[0][:limit]
    if mask.size == 0:
        raise SystemExit(f"No validation rows found in cache: {cache_path}")

    x_norm = ((x[mask] - mean) / std).astype(np.float32)
    q_features, input_scale = quantize_symmetric(x_norm)
    with torch.no_grad():
        logits = model(torch.from_numpy(x_norm))
        preds = logits.argmax(dim=1).cpu().numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["expected_class", "python_predicted_class"] + [f"x{i}" for i in range(490)])
        for expected, predicted, features in zip(y[mask], preds, q_features):
            writer.writerow([int(expected), int(predicted)] + [int(v) for v in features])

    params_path = path.with_suffix(".json")
    with params_path.open("w", encoding="utf-8") as f:
        json.dump({"input_scale": input_scale, "rows": int(mask.size)}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/checkpoints/kws_mlp.pt")
    parser.add_argument("--out", default="models/exported")
    parser.add_argument("--cache", help="Optional MFCC cache used to export golden vectors")
    parser.add_argument("--vectors", type=int, default=128)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = KwsMlp()
    model.load_state_dict(checkpoint["model"])
    state = model.state_dict()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quant_params = {
        "classes": CLASSES,
        "feature_mean_present": "feature_mean" in checkpoint,
        "feature_std_present": "feature_std" in checkpoint,
        "layers": [],
    }
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
    if args.cache:
        write_vector_csv(out / "golden_vectors.csv", model, checkpoint, Path(args.cache), args.vectors)
    print(f"Exported fixed-point artifacts to {out}")


if __name__ == "__main__":
    main()
