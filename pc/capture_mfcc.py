#!/usr/bin/env python3
"""Capture or load audio and write a 49x10 MFCC feature window.

This implementation intentionally avoids librosa/numba because some Windows
lab machines block numba DLL loading. It uses only NumPy and SciPy.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import resample_poly


SAMPLE_RATE = 16_000
DEFAULT_RECORD_RATE = 44_100
N_MFCC = 10
N_FRAMES = 49
N_FFT = 512
WIN_LENGTH = 480
HOP_LENGTH = 320
N_MELS = 40


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int) -> np.ndarray:
    mel_min = _hz_to_mel(np.array([0.0]))[0]
    mel_max = _hz_to_mel(np.array([sample_rate / 2]))[0]
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((N_FFT + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            filters[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


def _to_float_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        channel_rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2, axis=0))
        audio = audio[:, int(np.argmax(channel_rms))]
    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_value
    return audio.astype(np.float32)


def load_wav(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    sr, audio = wavfile.read(path)
    audio = _to_float_mono(audio)
    if sr != sample_rate:
        gcd = np.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // gcd, sr // gcd).astype(np.float32)
    return audio


def record_audio_window(
    seconds: float,
    samplerate: int = DEFAULT_RECORD_RATE,
    channels: int = 1,
    device: int | None = None,
    blocksize: int = 1024,
) -> np.ndarray:
    """Record one audio window using the continuous InputStream style."""
    chunks: list[np.ndarray] = []
    target_samples = int(seconds * samplerate)
    captured_samples = 0

    def callback(indata, frames, tinfo, status) -> None:
        del tinfo
        nonlocal captured_samples
        if status:
            print(f"audio status={status}", flush=True)
        if captured_samples < target_samples:
            chunks.append(indata.copy())
            captured_samples += frames

    with sd.InputStream(
        channels=channels,
        samplerate=samplerate,
        blocksize=blocksize,
        device=device,
        dtype="float32",
        callback=callback,
    ):
        deadline = time.time() + seconds + 2.0
        while captured_samples < target_samples and time.time() < deadline:
            time.sleep(0.01)

    if not chunks:
        raise RuntimeError("No microphone samples captured")
    audio = np.concatenate(chunks, axis=0)[:target_samples]
    return audio.astype(np.float32)


def mfcc_from_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = _to_float_mono(np.asarray(audio))
    if sample_rate != SAMPLE_RATE:
        gcd = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // gcd, sample_rate // gcd).astype(np.float32)
        sample_rate = SAMPLE_RATE

    target_len = SAMPLE_RATE
    if audio.size < target_len:
        audio = np.pad(audio, (0, target_len - audio.size))
    audio = audio[:target_len]

    preemph = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    frames = []
    for start in range(0, target_len - WIN_LENGTH + 1, HOP_LENGTH):
        frame = preemph[start:start + WIN_LENGTH]
        frame = frame * np.hamming(WIN_LENGTH).astype(np.float32)
        frames.append(frame)
    frame_mat = np.stack(frames).astype(np.float32)

    spectrum = np.fft.rfft(frame_mat, n=N_FFT)
    power = (np.abs(spectrum) ** 2) / N_FFT
    mel_energy = np.maximum(power @ _mel_filterbank(sample_rate).T, 1e-10)
    log_mel = np.log(mel_energy)
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]

    if coeffs.shape[0] < N_FRAMES:
        pad = np.zeros((N_FRAMES - coeffs.shape[0], N_MFCC), dtype=np.float32)
        coeffs = np.vstack([coeffs, pad])
    return coeffs[:N_FRAMES].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", help="Optional WAV file; if omitted, record from microphone")
    parser.add_argument("--out", default="mfcc.csv", help="Output CSV path")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--samplerate", type=int, default=DEFAULT_RECORD_RATE)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--blocksize", type=int, default=1024)
    args = parser.parse_args()

    if args.wav:
        audio = load_wav(args.wav)
        audio_sample_rate = SAMPLE_RATE
    else:
        print(f"Recording {args.seconds:.1f}s at {args.channels}ch@{args.samplerate} Hz...")
        audio = record_audio_window(
            seconds=args.seconds,
            samplerate=args.samplerate,
            channels=args.channels,
            device=args.device,
            blocksize=args.blocksize,
        )
        mono = _to_float_mono(audio)
        rms = float(np.sqrt(np.mean(mono ** 2)))
        peak = float(np.max(np.abs(mono)))
        print(f"Captured rms={rms:.5f} peak={peak:.5f}")
        audio_sample_rate = args.samplerate

    features = mfcc_from_audio(audio, sample_rate=audio_sample_rate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, features, delimiter=",", fmt="%.7f")
    print(f"Wrote {features.shape[0]}x{features.shape[1]} MFCCs to {out}")


if __name__ == "__main__":
    main()
