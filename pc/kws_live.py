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


def load_model(path: Path) -> KwsMlp:
    checkpoint = torch.load(path, map_location="cpu")
    model = KwsMlp()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict(model: KwsMlp, features: np.ndarray) -> tuple[str, float]:
    x = torch.from_numpy(features.reshape(1, -1).astype(np.float32))
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    score, idx = torch.max(probs, dim=0)
    return CLASSES[int(idx)], float(score)


def run_live(args: argparse.Namespace) -> None:
    port = args.port or find_port()
    if not port:
        raise SystemExit("No Cmod A7 serial port found. Use --port COMxx.")

    model = None if args.fpga_inference else load_model(Path(args.model))
    ser = serial.Serial(port, args.baud, timeout=0)
    frame_index = 0
    last_fire = 0.0
    print(f"Serial: {port} @ {args.baud}")

    try:
        while True:
            audio = sd.rec(SAMPLE_RATE, samplerate=SAMPLE_RATE, channels=1,
                           dtype="float32", device=args.device)
            sd.wait()
            features = mfcc_from_audio(audio[:, 0])

            if args.fpga_inference:
                q = np.clip(np.rint(features / args.input_scale), -128, 127).astype(np.int8)
                for row in q:
                    send_mfcc_frame(ser, frame_index, row)
                    frame_index = (frame_index + 1) & 0xFF
                    time.sleep(0.02)
            else:
                label, score = predict(model, features)
                now = time.time()
                if score >= args.threshold and now - last_fire >= args.refractory:
                    last_fire = now
                    send_led_frame(ser, CLASS_TO_LED[label])
                elif label == "silence":
                    send_led_frame(ser, CLASS_TO_LED["silence"])
                print(f"{label:8s} score={score:.3f}")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "vectors"], default="live")
    parser.add_argument("--model", default="models/checkpoints/kws_mlp.pt")
    parser.add_argument("--vectors", default="models/exported/golden_vectors.csv")
    parser.add_argument("--fpga-inference", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--refractory", type=float, default=1.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == "vectors":
        run_vectors(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
