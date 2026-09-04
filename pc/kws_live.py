#!/usr/bin/env python3
"""Phase 1/2 keyword spotting runner.

Phase 1 mode runs laptop-side classification and sends LED frames.
Phase 2 mode streams MFCC feature frames for FPGA-side inference.
"""
import argparse
import csv
import json
import queue
import time
from pathlib import Path

import numpy as np
import serial
import serial.tools.list_ports as list_ports
import torch

from capture_mfcc import DEFAULT_RECORD_RATE, N_FEATURES, SAMPLE_RATE, mfcc_from_audio
from capture_mfcc import load_wav
from train_mlp import CLASSES, KwsMlp


LED_SYNC = 0xFF
MFCC_SYNC = 0xFE
CLASS_REPLY_SYNC = 0xCA
FRAMES_PER_WINDOW = 49
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
    coeffs = np.asarray(coeffs, dtype=np.int8).reshape(N_FEATURES)
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


def load_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the training feature mean/std from a checkpoint without building the
    model. Needed by the FPGA-inference path, which quantizes on the PC but runs
    the network on-chip."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mean = checkpoint.get("feature_mean", np.zeros(FRAMES_PER_WINDOW * N_FEATURES, dtype=np.float32))
    std = checkpoint.get("feature_std", np.ones(FRAMES_PER_WINDOW * N_FEATURES, dtype=np.float32))
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def load_input_scale(path: Path) -> float:
    """Read the int8 input scale the FPGA was calibrated on (quant_params.json)."""
    with open(path, encoding="utf-8") as f:
        params = json.load(f)
    return float(params["input_scale"])


def quantize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray,
                      input_scale: float) -> np.ndarray:
    """Quantize a 49x20 MFCC window to int8 exactly as export_weights.py did for
    the golden vectors: normalize with training stats, divide by input_scale,
    round to nearest (np.rint, ties to even), clamp to [-128, 127]. This is the
    numeric contract the FPGA was verified against, so live windows must be
    quantized the same way."""
    flat = features.reshape(-1).astype(np.float32)
    norm = (flat - mean) / std
    q = np.clip(np.rint(norm / input_scale), -128, 127).astype(np.int8)
    return q.reshape(FRAMES_PER_WINDOW, N_FEATURES)


def predict_probs(model: KwsMlp, mean: np.ndarray, std: np.ndarray, features: np.ndarray) -> np.ndarray:
    flat = features.reshape(-1).astype(np.float32)
    x = torch.from_numpy(((flat - mean) / std).reshape(1, -1).astype(np.float32))
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return probs.numpy().astype(np.float32)


def predict(model: KwsMlp, mean: np.ndarray, std: np.ndarray, features: np.ndarray) -> tuple[str, float]:
    probs = predict_probs(model, mean, std, features)
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])


def _audio_windows(args: argparse.Namespace, drain_to_latest: bool = False):
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    target_samples = int(args.window_seconds * args.samplerate)
    hop_samples = max(1, int(args.hop_seconds * args.samplerate))

    def callback(indata, frames, tinfo, status) -> None:
        del frames, tinfo
        if status:
            print(f"audio status={status}", flush=True)
        audio_queue.put(indata.copy())

    import sounddevice as sd

    print(
        "Audio: "
        f"device={args.device if args.device is not None else 'default'} "
        f"{args.channels}ch@{args.samplerate}, blocksize={args.blocksize}, "
        f"window={args.window_seconds}s hop={args.hop_seconds}s",
        flush=True,
    )
    with sd.InputStream(
        channels=args.channels,
        samplerate=args.samplerate,
        blocksize=args.blocksize,
        device=args.device,
        dtype="float32",
        callback=callback,
    ):
        carry = np.empty((0, args.channels), dtype=np.float32)
        while True:
            while carry.shape[0] < target_samples:
                carry = np.concatenate([carry, audio_queue.get()], axis=0)
            if drain_to_latest:
                # When the consumer is slower than real time (e.g. the FPGA serial
                # round-trip), pull every already-captured chunk and keep only the
                # newest window, so classifications never lag behind live speech.
                try:
                    while True:
                        carry = np.concatenate([carry, audio_queue.get_nowait()], axis=0)
                except queue.Empty:
                    pass
                carry = carry[-target_samples:]
            # Overlapping windows: emit a full window, then advance by hop only, so a
            # spoken word is captured well-centered in at least one window.
            yield carry[:target_samples]
            carry = carry[hop_samples:]


def run_live(args: argparse.Namespace) -> None:
    port = args.port or find_port()
    if not port:
        raise SystemExit("No Cmod A7 serial port found. Use --port COMxx.")

    # FPGA-inference path quantizes on the PC but classifies + drives LEDs
    # on-chip; it needs the training stats (mean/std) and the calibrated int8
    # input scale, not the torch model.
    if args.fpga_inference:
        model_bundle = None
        fpga_mean, fpga_std = load_norm_stats(Path(args.model))
        input_scale = load_input_scale(Path(args.quant_params))
        print(f"FPGA inference: input_scale={input_scale:.6g} from {args.quant_params}", flush=True)
    else:
        model_bundle = load_model(Path(args.model))
        fpga_mean = fpga_std = None
        input_scale = 0.0

    # A small read timeout lets the FPGA class reply be polled without a busy spin.
    ser = serial.Serial(port, args.baud, timeout=0.05 if args.fpga_inference else 0)
    last_fire = 0.0
    smoothed = np.zeros(len(CLASSES), dtype=np.float32)  # EMA of class probabilities
    print(f"Serial: {port} @ {args.baud}", flush=True)

    try:
        for audio in _audio_windows(args, drain_to_latest=args.fpga_inference):
            if audio.ndim > 1:
                channel_rms = np.sqrt(np.mean(audio ** 2, axis=0))
                mono = audio[:, int(np.argmax(channel_rms))]
            else:
                mono = audio
            rms = float(np.sqrt(np.mean(mono ** 2)))
            peak = float(np.max(np.abs(mono)))
            features = mfcc_from_audio(audio, sample_rate=args.samplerate)

            if args.fpga_inference:
                # Optionally hold on quiet audio so the FPGA keeps its last class
                # instead of classifying near-silence noise.
                if rms < args.min_rms:
                    print(f"quiet    rms={rms:.5f} peak={peak:.5f} (window held)", flush=True)
                    continue
                frames = quantize_features(features, fpga_mean, fpga_std, input_scale)
                ser.reset_input_buffer()
                for frame_index in range(FRAMES_PER_WINDOW):
                    send_mfcc_frame(ser, frame_index, frames[frame_index])
                    time.sleep(args.frame_delay)
                # The FPGA classifies the 49-frame window, drives its own LEDs via
                # class_to_led, and echoes the class over the 0xCA reply frame.
                got = read_class_reply(ser, timeout=args.reply_timeout)
                label = CLASSES[got] if got is not None and 0 <= got < len(CLASSES) else "?"
                fpga_str = "none" if got is None else str(got)
                print(f"FPGA={label:8s} class={fpga_str:>4s} rms={rms:.5f} peak={peak:.5f}",
                      flush=True)
            else:
                model, mean, std = model_bundle
                if rms < args.min_rms:
                    probs = np.zeros(len(CLASSES), dtype=np.float32)
                    probs[CLASSES.index("silence")] = 1.0
                else:
                    probs = predict_probs(model, mean, std, features)
                # Temporal smoothing: EMA over overlapping windows to suppress flicker.
                smoothed = args.smooth * smoothed + (1.0 - args.smooth) * probs
                idx = int(np.argmax(smoothed))
                label, score = CLASSES[idx], float(smoothed[idx])

                now = time.time()
                if label != "silence" and score >= args.threshold and now - last_fire >= args.refractory:
                    last_fire = now
                    send_led_frame(ser, CLASS_TO_LED[label])
                elif label == "silence":
                    send_led_frame(ser, CLASS_TO_LED["silence"])
                print(f"{label:8s} score={score:.3f} rms={rms:.5f} peak={peak:.5f}", flush=True)
    except KeyboardInterrupt:
        # In FPGA mode the board drives its own LEDs; there is no PC LED frame to send.
        if not args.fpga_inference:
            send_led_frame(ser, CLASS_TO_LED["silence"])
    finally:
        ser.close()


def read_class_reply(ser: serial.Serial, timeout: float) -> int | None:
    """Read one 0xCA class-reply frame (0xCA, class_id, checksum).

    Returns the class id, or None on timeout / bad checksum.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b[0] != CLASS_REPLY_SYNC:
            continue  # hunt for sync
        rest = ser.read(2)
        if len(rest) < 2:
            return None
        class_id, chk = rest[0], rest[1]
        if ((CLASS_REPLY_SYNC + class_id) & 0xFF) != chk:
            return None  # corrupt reply
        return class_id
    return None


def run_vectors(args: argparse.Namespace) -> None:
    """Stream golden feature windows to the FPGA and check its predicted class.

    Each CSV row is one full 49x20 = 980 feature window. The row is split into
    49 MFCC frames and streamed; the FPGA replies with its predicted class,
    which is compared against the Python fixed-point reference
    (fp_predicted_class, column index 1). This is the on-hardware analogue of
    the Phase 2B simulation gate.
    """
    port = args.port or find_port()
    if not port:
        raise SystemExit("No Cmod A7 serial port found. Use --port COMxx.")
    ser = serial.Serial(port, args.baud, timeout=0.05)
    time.sleep(0.2)
    ser.reset_input_buffer()

    total = 0
    passed = 0
    mismatches: list[tuple[int, int, int | None]] = []
    with open(args.vectors, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header row
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if args.limit and total >= args.limit:
                break
            expected_fp = int(row[1])  # fp_predicted_class
            feats = np.array([int(v) for v in row[3:3 + FRAMES_PER_WINDOW * N_FEATURES]],
                             dtype=np.int8)
            if feats.size != FRAMES_PER_WINDOW * N_FEATURES:
                print(f"skip: row {total + 1} has {feats.size} features, "
                      f"expected {FRAMES_PER_WINDOW * N_FEATURES}")
                continue
            frames = feats.reshape(FRAMES_PER_WINDOW, N_FEATURES)

            ser.reset_input_buffer()
            for frame_index in range(FRAMES_PER_WINDOW):
                send_mfcc_frame(ser, frame_index, frames[frame_index])
                time.sleep(args.frame_delay)

            got = read_class_reply(ser, timeout=args.reply_timeout)
            total += 1
            ok = got is not None and got == expected_fp
            passed += int(ok)
            got_str = "none" if got is None else str(got)
            print(f"{'pass' if ok else 'fail'} vector={total:3d} "
                  f"expected={expected_fp:2d} fpga={got_str}")
            if not ok:
                mismatches.append((total, expected_fp, got))
    ser.close()

    print(f"\nPhase 2C: {passed}/{total} vectors match the FPGA class")
    if mismatches:
        print(f"mismatches (vector, expected, fpga): {mismatches[:20]}")


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
    parser.add_argument("--quant-params", default="models/exported/quant_params.json",
                        help="FPGA int8 input_scale source for --fpga-inference live mode")
    parser.add_argument("--wav-dir", default="data/local_test_wavs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--send-leds", action="store_true")
    parser.add_argument("--fpga-inference", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--samplerate", type=int, default=DEFAULT_RECORD_RATE)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--hop-seconds", type=float, default=0.25)
    parser.add_argument("--smooth", type=float, default=0.6, help="EMA factor for probability smoothing [0,1)")
    parser.add_argument("--min-rms", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--refractory", type=float, default=1.0)
    parser.add_argument("--frame-delay", type=float, default=0.01,
                        help="seconds between MFCC frames in vectors mode")
    parser.add_argument("--reply-timeout", type=float, default=2.0,
                        help="seconds to wait for the FPGA class reply in vectors mode")
    args = parser.parse_args()

    if args.mode == "vectors":
        run_vectors(args)
    elif args.mode == "wav-dir":
        run_wav_dir(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
