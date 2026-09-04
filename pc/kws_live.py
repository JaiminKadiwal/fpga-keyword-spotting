#!/usr/bin/env python3
"""Phase 1/2 keyword spotting runner.

Phase 1 mode runs laptop-side classification and sends LED frames.
Phase 2 mode streams MFCC feature frames for FPGA-side inference.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports as list_ports
import sounddevice as sd
import torch

from capture_mfcc import SAMPLE_RATE, mfcc_from_audio
from capture_mfcc import load_wav
from train_mlp import CLASSES, KwsMlp


LED_SYNC = 0xFF
MFCC_SYNC = 0xFE
FTDI_VIDPID = (0x0403, 0x6010)

CLASS_TO_LED = {
    "silence": [0, 0, 0, 0, 0],
    "unknown": [0, 0, 40, 40, 40],
    "yes": [128, 0, 0, 254, 0],
    "no": [0, 128, 254, 0, 0],
    "up": [0, 0, 0, 0, 254],
    "down": [0, 0, 0, 254, 254],
    "left": [0, 0, 128, 0, 254],
    "right": [0, 0, 254, 128, 0],
    "on": [128, 0, 128, 254, 0],
    "off": [0, 0, 0, 0, 128],
    "stop": [254, 254, 254, 0, 0],
    "go": [128, 0, 0, 254, 128],
}


def find_port() -> str | None:
    for port in list_ports.comports():
        if (port.vid, port.pid) == FTDI_VIDPID:
            return port.device
    for port in list_ports.comports():
        if "USB Serial" in (port.description or ""):
            return port.device
    return None


def send_led_frame(ser: serial.Serial, channels: list[int]) -> None:
    ser.write(bytes([LED_SYNC] + [max(0, min(254, int(v))) for v in channels]))


def send_mfcc_frame(ser: serial.Serial, frame_index: int, coeffs: np.ndarray) -> None:
    coeffs = np.asarray(coeffs, dtype=np.int8).reshape(10)
    payload = [frame_index & 0xFF] + [(int(v) & 0xFF) for v in coeffs]
    checksum = sum(payload) & 0xFF
    ser.write(bytes([MFCC_SYNC] + payload + [checksum]))


def load_model(path: Path) -> tuple[KwsMlp, np.ndarray, np.ndarray]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = KwsMlp()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = checkpoint.get("feature_mean", np.zeros(490, dtype=np.float32))
    std = checkpoint.get("feature_std", np.ones(490, dtype=np.float32))
    return model, np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def predict(model: KwsMlp, mean: np.ndarray, std: np.ndarray, features: np.ndarray) -> tuple[str, float]:
    flat = features.reshape(-1).astype(np.float32)
    x = torch.from_numpy(((flat - mean) / std).reshape(1, -1).astype(np.float32))
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    score, idx = torch.max(probs, dim=0)
    return CLASSES[int(idx)], float(score)


def run_live(args: argparse.Namespace) -> None:
    port = args.port or find_port()
    if not port:
        raise SystemExit("No Cmod A7 serial port found. Use --port COMxx.")

    model_bundle = None if args.fpga_inference else load_model(Path(args.model))
    ser = serial.Serial(port, args.baud, timeout=0)
    frame_index = 0
    last_fire = 0.0
    print(f"Serial: {port} @ {args.baud}", flush=True)

    try:
        while True:
            samples = int(args.window_seconds * args.samplerate)
            audio = sd.rec(samples, samplerate=args.samplerate, channels=args.channels,
                           dtype="float32", device=args.device)
            sd.wait()
            if audio.ndim > 1:
                channel_rms = np.sqrt(np.mean(audio ** 2, axis=0))
                mono = audio[:, int(np.argmax(channel_rms))]
            else:
                mono = audio
            rms = float(np.sqrt(np.mean(mono ** 2)))
            peak = float(np.max(np.abs(mono)))
            features = mfcc_from_audio(audio, sample_rate=args.samplerate)

            if args.fpga_inference:
                q = np.clip(np.rint(features / args.input_scale), -128, 127).astype(np.int8)
                for row in q:
                    send_mfcc_frame(ser, frame_index, row)
                    frame_index = (frame_index + 1) & 0xFF
                    time.sleep(0.02)
            else:
                model, mean, std = model_bundle
                if rms < args.min_rms:
                    label, score = "silence", 1.0
                else:
                    label, score = predict(model, mean, std, features)
                now = time.time()
                if score >= args.threshold and now - last_fire >= args.refractory:
                    last_fire = now
                    send_led_frame(ser, CLASS_TO_LED[label])
                elif label == "silence":
                    send_led_frame(ser, CLASS_TO_LED["silence"])
                print(f"{label:8s} score={score:.3f} rms={rms:.5f} peak={peak:.5f}", flush=True)
    except KeyboardInterrupt:
        send_led_frame(ser, CLASS_TO_LED["silence"])
    finally:
        ser.close()


def run_vectors(args: argparse.Namespace) -> None:
    port = args.port or find_port()
    if not port:
        raise SystemExit("No Cmod A7 serial port found. Use --port COMxx.")
    ser = serial.Serial(port, args.baud, timeout=0)
    frame_index = 0
    with open(args.vectors, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            coeffs = np.array([int(v) for v in row[:10]], dtype=np.int8)
            send_mfcc_frame(ser, frame_index, coeffs)
            frame_index = (frame_index + 1) & 0xFF
            time.sleep(0.02)
    ser.close()


def _label_from_path(path: Path) -> str | None:
    parent_label = path.parent.name
    if parent_label in CLASSES:
        return parent_label
    stem_label = path.stem.split("_")[0].lower()
    return stem_label if stem_label in CLASSES else None


def run_wav_dir(args: argparse.Namespace) -> None:
    model, mean, std = load_model(Path(args.model))
    wav_dir = Path(args.wav_dir)
    wavs = sorted(wav_dir.rglob("*.wav"))
    if args.limit:
        wavs = wavs[:args.limit]
    if not wavs:
        raise SystemExit(f"No .wav files found under {wav_dir}")

    port = args.port or find_port()
    ser = serial.Serial(port, args.baud, timeout=0) if port and args.send_leds else None
    correct = 0
    counted = 0
    try:
        for wav_path in wavs:
            expected = _label_from_path(wav_path)
            audio = load_wav(wav_path)
            features = mfcc_from_audio(audio)
            label, score = predict(model, mean, std, features)
            if ser is not None and score >= args.threshold:
                send_led_frame(ser, CLASS_TO_LED[label])
                time.sleep(args.refractory)

            result = "skip"
            if expected is not None:
                counted += 1
                correct += int(label == expected)
                result = "pass" if label == expected else "fail"
            print(f"{result:4s} expected={expected or '-':8s} predicted={label:8s} score={score:.3f} file={wav_path}")
    finally:
        if ser is not None:
            send_led_frame(ser, CLASS_TO_LED["silence"])
            ser.close()

    if counted:
        print(f"Accuracy: {correct}/{counted} = {100.0 * correct / counted:.1f}%")
    else:
        print("No class labels inferred from paths; place WAVs in class-named folders to score accuracy.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "vectors", "wav-dir"], default="live")
    parser.add_argument("--model", default="models/checkpoints/kws_mlp.pt")
    parser.add_argument("--vectors", default="models/exported/golden_vectors.csv")
    parser.add_argument("--wav-dir", default="data/local_test_wavs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--send-leds", action="store_true")
    parser.add_argument("--fpga-inference", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--samplerate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--min-rms", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--refractory", type=float, default=1.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == "vectors":
        run_vectors(args)
    elif args.mode == "wav-dir":
        run_wav_dir(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
