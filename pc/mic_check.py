#!/usr/bin/env python3
"""List microphone devices and print live input levels."""
import argparse
import time

import numpy as np
import sounddevice as sd


def list_devices() -> None:
    hostapis = sd.query_hostapis()
    print("Input devices:")
    for i, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            hostapi = hostapis[device["hostapi"]]["name"]
            default = " <-- default" if i == sd.default.device[0] else ""
            print(
                f"{i:2d}: {device['name']} | {hostapi} | "
                f"inputs={device['max_input_channels']} | "
                f"default_sr={device['default_samplerate']}{default}"
            )


def probe_devices() -> None:
    sample_rates = [16000, 44100, 48000]
    channel_counts = [1, 2, 4]
    print("Openable input formats:")
    for i, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] <= 0:
            continue
        ok: list[str] = []
        for channels in channel_counts:
            if channels > device["max_input_channels"]:
                continue
            for sample_rate in sample_rates:
                try:
                    sd.check_input_settings(device=i, channels=channels, samplerate=sample_rate, dtype="float32")
                    ok.append(f"{channels}ch@{sample_rate}")
                except Exception:
                    pass
        print(f"{i:2d}: {device['name']} -> {', '.join(ok) if ok else 'no checked format opened'}")


def meter(device: int | None, seconds: float, samplerate: int, channels: int, blocksize: int) -> None:
    print(
        f"Metering device={device if device is not None else 'default'} "
        f"for {seconds:.1f}s at {channels}ch@{samplerate}"
    )
    print("Speak near the laptop. RMS/peak should jump clearly if the mic is live.")
    start = time.time()
    rms_values: list[float] = []
    peak_values: list[float] = []

    def callback(indata, frames, tinfo, status) -> None:
        del frames, tinfo
        if status:
            print(f"status={status}", flush=True)
        if indata.ndim > 1:
            channel_rms = np.sqrt(np.mean(indata ** 2, axis=0))
            mono = indata[:, int(np.argmax(channel_rms))]
            channel_text = " ch_rms=" + ",".join(f"{v:.5f}" for v in channel_rms)
        else:
            mono = indata
            channel_text = ""
        rms = float(np.sqrt(np.mean(mono ** 2)))
        peak = float(np.max(np.abs(mono)))
        rms_values.append(rms)
        peak_values.append(peak)
        bars = "#" * min(60, int(rms * 600))
        print(f"rms={rms:.5f} peak={peak:.5f} |{bars}{channel_text}", flush=True)

    with sd.InputStream(
        channels=channels,
        samplerate=samplerate,
        blocksize=blocksize,
        device=device,
        dtype="float32",
        callback=callback,
    ):
        while time.time() - start < seconds:
            time.sleep(0.1)

    if rms_values:
        rms_arr = np.asarray(rms_values)
        peak_arr = np.asarray(peak_values)
        print(
            "Summary: "
            f"rms_min={rms_arr.min():.5f} "
            f"rms_med={np.median(rms_arr):.5f} "
            f"rms_max={rms_arr.max():.5f} "
            f"peak_max={peak_arr.max():.5f}"
        )
        if rms_arr.max() < 0.01:
            print("Result: input opened, but speech was not clearly detected.")
        else:
            print("Result: microphone signal is active.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--blocksize", type=int, default=1024)
    args = parser.parse_args()

    if args.list:
        list_devices()
    elif args.probe:
        probe_devices()
    else:
        meter(args.device, args.seconds, args.samplerate, args.channels, args.blocksize)


if __name__ == "__main__":
    main()
