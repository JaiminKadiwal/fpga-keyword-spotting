#!/usr/bin/env python3
"""Train the first-pass MLP for Speech Commands KWS."""
import argparse
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from capture_mfcc import SAMPLE_RATE, mfcc_from_audio


CLASSES = [
    "silence", "unknown", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go",
]
TARGET_WORDS = set(CLASSES[2:])


class KwsMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(490, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpeechCommandsMfcc(Dataset):
    def __init__(self, root: Path, split: str, max_items: int | None = None) -> None:
        self.root = root
        self.items = self._collect_items(root, split)
        if max_items:
            random.Random(7).shuffle(self.items)
            self.items = self.items[:max_items]

    def _collect_items(self, root: Path, split: str) -> list[tuple[Path | None, int]]:
        all_wavs = [p for p in root.glob("*/*.wav") if "_background_noise_" not in p.parts]
        random.Random(13).shuffle(all_wavs)
        n = len(all_wavs)
        if split == "train":
            selected = all_wavs[: int(0.8 * n)]
        elif split == "val":
            selected = all_wavs[int(0.8 * n): int(0.9 * n)]
        else:
            selected = all_wavs[int(0.9 * n):]

        items: list[tuple[Path | None, int]] = []
        for wav in selected:
            word = wav.parent.name
            label = CLASSES.index(word) if word in TARGET_WORDS else CLASSES.index("unknown")
            items.append((wav, label))

        silence_count = max(1, len(items) // 20)
        items.extend([(None, CLASSES.index("silence")) for _ in range(silence_count)])
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav, label = self.items[index]
        if wav is None:
            features = np.zeros((49, 10), dtype=np.float32)
        else:
            audio, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
            features = mfcc_from_audio(audio)
        x = torch.from_numpy(features.reshape(-1))
        y = torch.tensor(label, dtype=torch.long)
        return x, y


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--out", default="models/checkpoints/kws_mlp.pt")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise SystemExit(f"Dataset not found: {root}")

    train_ds = SpeechCommandsMfcc(root, "train", args.max_items)
    val_ds = SpeechCommandsMfcc(root, "val", args.max_items)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = KwsMlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x, y in progress:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{float(loss):.4f}")
        acc = evaluate(model, val_loader)
        print(f"val_acc={acc:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": CLASSES}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
