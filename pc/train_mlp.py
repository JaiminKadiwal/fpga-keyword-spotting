#!/usr/bin/env python3
"""Train the first-pass MLP for Speech Commands KWS.

The training script caches MFCC tensors to an ignored local NPZ file. That
makes repeated training runs practical on a laptop and keeps GitHub clean.
"""
import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from capture_mfcc import (
    FEATURE_DIM,
    SAMPLE_RATE,
    load_wav,
    mfcc_from_audio,
)


CLASSES = [
    "silence", "unknown", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go",
]
TARGET_WORDS = set(CLASSES[2:])

TRAIN_AUG_COPIES = 2          # augmented variants emitted per train clip (plus the clean one)
AUG_MAX_SHIFT = SAMPLE_RATE // 10  # +/-100 ms time shift
AUG_SNR_DB = (10.0, 25.0)     # background-noise mix SNR range (milder = closer to clean val)


class KwsMlp(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class Item:
    wav: Path | None
    label: int
    split: str


class SpeechCommandsIndex:
    def __init__(
        self,
        root: Path,
        max_items: int | None,
        max_per_class: int | None,
        unknown_mult: int = 3,
    ) -> None:
        self.root = root
        self.max_items = max_items
        self.max_per_class = max_per_class
        # "unknown" spans ~20 non-target words, so give it more clips than a single word.
        self.unknown_mult = unknown_mult

    def items(self) -> list[Item]:
        all_wavs = [p for p in self.root.glob("*/*.wav") if "_background_noise_" not in p.parts]
        if not all_wavs:
            raise SystemExit(f"No WAV files found under {self.root}. Run pc/prepare_speech_commands.py first.")

        rng = random.Random(13)
        by_label: dict[int, list[Path]] = {i: [] for i in range(len(CLASSES))}
        for wav in all_wavs:
            word = wav.parent.name
            label = CLASSES.index(word) if word in TARGET_WORDS else CLASSES.index("unknown")
            by_label[label].append(wav)

        unknown_idx = CLASSES.index("unknown")
        wavs: list[Path] = []
        for label, paths in by_label.items():
            if label == CLASSES.index("silence"):
                continue
            rng.shuffle(paths)
            cap = self.max_per_class
            if cap and label == unknown_idx:
                cap = cap * self.unknown_mult
            wavs.extend(paths[:cap] if cap else paths)

        rng.shuffle(wavs)
        if self.max_items:
            wavs = wavs[: self.max_items]

        items: list[Item] = []
        n = len(wavs)
        for i, wav in enumerate(wavs):
            if i < int(0.8 * n):
                split = "train"
            elif i < int(0.9 * n):
                split = "val"
            else:
                split = "test"
            word = wav.parent.name
            label = CLASSES.index(word) if word in TARGET_WORDS else CLASSES.index("unknown")
            items.append(Item(wav, label, split))

        for split in ["train", "val", "test"]:
            split_count = sum(1 for item in items if item.split == split)
            silence_count = max(1, split_count // 20)
            items.extend(Item(None, CLASSES.index("silence"), split) for _ in range(silence_count))
        return items


def _fixed_length(audio: np.ndarray, length: int = SAMPLE_RATE) -> np.ndarray:
    if audio.size < length:
        return np.pad(audio, (0, length - audio.size)).astype(np.float32)
    return audio[:length].astype(np.float32)


def _load_noise_pool(root: Path) -> list[np.ndarray]:
    noise_dir = root / "_background_noise_"
    pool: list[np.ndarray] = []
    if noise_dir.is_dir():
        for wav in sorted(noise_dir.glob("*.wav")):
            pool.append(load_wav(wav))  # resampled to 16 kHz mono
    return pool


def _random_noise_crop(pool: list[np.ndarray], rng: random.Random, length: int = SAMPLE_RATE) -> np.ndarray:
    if not pool:
        return np.zeros(length, dtype=np.float32)
    clip = pool[rng.randrange(len(pool))]
    if clip.size <= length:
        return _fixed_length(clip, length)
    start = rng.randrange(clip.size - length)
    return clip[start:start + length].astype(np.float32)


def _augment(audio: np.ndarray, pool: list[np.ndarray], rng: random.Random) -> np.ndarray:
    """Time-shift + background-noise mix. Gain is intentionally omitted because the
    MFCC front-end peak-normalizes each window, which would cancel pure gain changes."""
    shift = rng.randint(-AUG_MAX_SHIFT, AUG_MAX_SHIFT)
    shifted = np.zeros_like(audio)
    if shift >= 0:
        shifted[shift:] = audio[: audio.size - shift]
    else:
        shifted[:shift] = audio[-shift:]

    noise = _random_noise_crop(pool, rng, audio.size)
    sig_power = float(np.mean(shifted ** 2)) + 1e-10
    noise_power = float(np.mean(noise ** 2)) + 1e-10
    snr_db = rng.uniform(*AUG_SNR_DB)
    scale = math.sqrt(sig_power / ((10.0 ** (snr_db / 10.0)) * noise_power))
    return (shifted + scale * noise).astype(np.float32)


class MfccCacheBuilder:
    def __init__(self, items: list[Item], root: Path, aug_copies: int = TRAIN_AUG_COPIES) -> None:
        self.items = items
        self.root = root
        self.aug_copies = aug_copies

    def build(self) -> dict[str, np.ndarray]:
        rng = random.Random(29)
        noise_pool = _load_noise_pool(self.root)
        xs: list[np.ndarray] = []
        ys: list[int] = []
        splits: list[str] = []

        def emit(features: np.ndarray, label: int, split: str) -> None:
            xs.append(features.reshape(-1))
            ys.append(label)
            splits.append(split)

        for item in tqdm(self.items, desc="MFCC cache"):
            if item.wav is None:
                # Silence = a real background-noise crop (falls back to zeros if none).
                base = _random_noise_crop(noise_pool, rng)
            else:
                base = _fixed_length(load_wav(item.wav))
            emit(mfcc_from_audio(base), item.label, item.split)

            # Augment train split only; val/test stay clean for honest evaluation.
            if item.split == "train" and self.aug_copies > 0 and noise_pool:
                for _ in range(self.aug_copies):
                    emit(mfcc_from_audio(_augment(base, noise_pool, rng)), item.label, item.split)

        return {
            "x": np.stack(xs).astype(np.float32),
            "y": np.array(ys, dtype=np.int64),
            "split": np.array(splits),
            "classes": np.array(CLASSES),
        }


def load_or_build_cache(
    root: Path,
    cache: Path,
    max_items: int | None,
    max_per_class: int | None,
    rebuild: bool,
) -> dict[str, np.ndarray]:
    if cache.exists() and not rebuild:
        print(f"Loading MFCC cache: {cache}")
        return dict(np.load(cache, allow_pickle=False))
    items = SpeechCommandsIndex(root, max_items=max_items, max_per_class=max_per_class).items()
    data = MfccCacheBuilder(items, root).build()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    print(f"Wrote MFCC cache: {cache}")
    return data


def smoke_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    splits: list[str] = []
    for label in range(len(CLASSES)):
        center = np.zeros(FEATURE_DIM, dtype=np.float32)
        center[label * 8: label * 8 + 8] = 12.0
        for i in range(12):
            xs.append(center + rng.normal(0, 0.6, FEATURE_DIM).astype(np.float32))
            ys.append(label)
            splits.append("train" if i < 8 else "val")
    return {
        "x": np.stack(xs),
        "y": np.array(ys, dtype=np.int64),
        "split": np.array(splits),
        "classes": np.array(CLASSES),
    }


def make_loader(
    data: dict[str, np.ndarray],
    split: str,
    batch_size: int,
    shuffle: bool,
    balanced: bool = False,
) -> DataLoader:
    mask = data["split"] == split
    x = torch.from_numpy(data["x"][mask].astype(np.float32))
    y = torch.from_numpy(data["y"][mask].astype(np.int64))
    dataset = TensorDataset(x, y)
    if balanced:
        labels = y.numpy()
        counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
        inv = 1.0 / np.maximum(counts, 1.0)
        sample_weights = inv[labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(labels),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def normalize_from_train(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    train_mask = data["split"] == "train"
    mean = data["x"][train_mask].mean(axis=0).astype(np.float32)
    std = data["x"][train_mask].std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4)
    normalized = dict(data)
    normalized["x"] = ((data["x"] - mean) / std).astype(np.float32)
    return normalized, mean, std


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += int(y.numel())
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/speech_commands")
    parser.add_argument("--cache", default="data/mfcc_cache.npz")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-balance", action="store_true", help="Disable class-balanced sampling")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--out", default="models/checkpoints/kws_mlp.pt")
    parser.add_argument("--smoke", action="store_true", help="Train on tiny synthetic data to verify the code path")
    parser.add_argument("--progress", action="store_true", help="Show per-batch progress bars")
    args = parser.parse_args()

    if args.smoke:
        data = smoke_data()
    else:
        data = load_or_build_cache(
            Path(args.data),
            Path(args.cache),
            args.max_items,
            args.max_per_class,
            args.rebuild_cache,
        )
    data, feature_mean, feature_std = normalize_from_train(data)

    train_loader = make_loader(
        data, "train", args.batch_size, shuffle=True, balanced=not args.no_balance
    )
    val_loader = make_loader(data, "val", args.batch_size, shuffle=False)

    model = KwsMlp(dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        iterator = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}") if args.progress else train_loader
        for x, y in iterator:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
            if args.progress:
                iterator.set_postfix(loss=f"{float(loss.detach()):.4f}")
        acc = evaluate(model, val_loader)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"epoch={epoch:03d} lr={lr_now:.2e} train_loss={total_loss / max(batches, 1):.4f} "
            f"val_acc={acc:.4f} best={best_acc:.4f}"
        )
        scheduler.step()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({
        "model": model.state_dict(),
        "classes": CLASSES,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "best_val_acc": best_acc,
    }, out)
    print(f"Saved {out} best_val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
