#!/usr/bin/env python3
"""Capture or load audio and write a 49x10 MFCC feature window."""
import argparse
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16_000
N_MFCC = 10
N_FRAMES = 49


def mfcc_from_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    target_len = sample_rate
    if audio.size < target_len:
        audio = np.pad(audio, (0, target_len - audio.size))
    audio = audio[:target_len]
    features = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=480,
        hop_length=320,
        win_length=480,
        center=False,
    ).T
    if features.shape[0] < N_FRAMES:
        pad = np.zeros((N_FRAMES - features.shape[0], N_MFCC), dtype=np.float32)
        features = np.vstack([features, pad])
    return features[:N_FRAMES].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", help="Optional WAV file; if omitted, record from microphone")
    parser.add_argument("--out", default="mfcc.csv", help="Output CSV path")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    if args.wav:
        audio, sr = librosa.load(args.wav, sr=SAMPLE_RATE, mono=True)
    else:
        print(f"Recording {args.seconds:.1f}s at {SAMPLE_RATE} Hz...")
        audio = sd.rec(int(args.seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                       dtype="float32", device=args.device)
        sd.wait()
        audio = audio[:, 0]
        sr = SAMPLE_RATE

    features = mfcc_from_audio(audio, sr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, features, delimiter=",", fmt="%.7f")
    print(f"Wrote {features.shape[0]}x{features.shape[1]} MFCCs to {out}")


if __name__ == "__main__":
    main()
