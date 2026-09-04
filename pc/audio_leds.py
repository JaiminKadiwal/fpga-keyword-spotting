#!/usr/bin/env python3
"""
Sound-reactive LED driver for the Cmod A7 visualizer (v2 - vivid HSV mapping).

Captures the laptop microphone and streams 5 brightness bytes to the FPGA.
Instead of blending bass/mid/treble into R/G/B (which washes out to white on
broadband music), it builds ONE saturated color:

    hue    <- where the energy is  (bass=red, mid=green, treble=blue)
    value  <- how loud it is       (quiet = dark, loud = bright)
    sat    <- kept high            (colors stay pure, never grey/white)

Green LEDs stay punchy: LD1 kicks with the bass, LD2 sparkles on treble.

Frame format (matches top.vhd): 0xFF ch0 ch1 ch2 ch3 ch4  (data 0..254)
    ch0=LD1  ch1=LD2  ch2=R  ch3=G  ch4=B

Usage:
    python audio_leds.py                 # auto-detect COM port + default mic
    python audio_leds.py --list
    python audio_leds.py --port COM8 --device 1
"""
import argparse, colorsys, sys, time
import numpy as np
import serial
import serial.tools.list_ports as list_ports
import sounddevice as sd

SYNC        = 0xFF
BANDS_HZ    = [(30, 250), (250, 2000), (2000, 9000)]   # bass, mid, treble
BAND_HUE    = np.array([0.00, 0.33, 0.66])             # red, green, blue
# Raw bass energy dwarfs treble, so calibrate before deciding hue, otherwise the
# color sits on red forever. Boost mid/treble so a bright drop can swing to blue.
# Tuned for "Levels"-style EDM played near the laptop mic.
BAND_GAIN   = np.array([1.0, 2.4, 4.5])
HUE_SHARP   = 2.2                                      # >1 makes the dominant band win the color
FTDI_VIDPID = (0x0403, 0x6010)                         # Cmod A7 FT2232H UART


def find_port():
    for p in list_ports.comports():
        if (p.vid, p.pid) == FTDI_VIDPID:
            return p.device
    for p in list_ports.comports():
        if "USB Serial" in (p.description or ""):
            return p.device
    return None


def list_everything():
    print("Serial ports:")
    for p in list_ports.comports():
        tag = "  <-- Cmod A7" if (p.vid, p.pid) == FTDI_VIDPID else ""
        print(f"  {p.device}: {p.description}{tag}")
    print("\nInput (microphone) devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            tag = "  <-- default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}{tag}")


def main():
    ap = argparse.ArgumentParser(description="Cmod A7 sound-reactive LED driver")
    ap.add_argument("--port", help="serial port (default: auto-detect Cmod A7)")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--samplerate", type=int, default=44100)
    ap.add_argument("--blocksize", type=int, default=1024)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--gate", type=float, default=0.012, help="silence threshold (RMS); raise if idle noise flickers it")
    ap.add_argument("--contrast", type=float, default=1.6, help="brightness curve; higher = punchier (darker quiets)")
    ap.add_argument("--sat", type=float, default=1.0, help="color saturation 0..1 (keep high for vivid colors)")
    ap.add_argument("--hue-smooth", type=float, default=0.16, help="0..1; lower = color drifts more slowly")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_everything()
        return

    port = args.port or find_port()
    if not port:
        print("No Cmod A7 serial port found. Use --list, then --port COMxx.")
        sys.exit(1)

    ser = serial.Serial(port, args.baud, timeout=0)
    print(f"Serial: {port} @ {args.baud}  |  mic: {args.device if args.device is not None else 'default'}")
    print("Play audio. Ctrl+C to stop.  (BTN0 = lamp test, BTN1 = idle rainbow)")

    freqs = np.fft.rfftfreq(args.blocksize, 1.0 / args.samplerate)
    band_idx = [np.where((freqs >= lo) & (freqs < hi))[0] for lo, hi in BANDS_HZ]
    window = np.hanning(args.blocksize).astype(np.float32)

    peaks     = np.array([1e-6, 1e-6, 1e-6])   # per-band AGC
    loud_peak = 1e-6                            # overall-loudness AGC
    hue = 0.33                                  # smoothed state
    val = 0.0
    g0  = 0.0
    g1  = 0.0
    last_print = 0.0

    def byte(x):
        return int(np.clip(round(x * 254), 0, 254))

    def callback(indata, frames, tinfo, status):
        nonlocal peaks, loud_peak, hue, val, g0, g1, last_print
        mono = indata[:, 0] if indata.ndim > 1 else indata
        if len(mono) < args.blocksize:
            mono = np.pad(mono, (0, args.blocksize - len(mono)))
        rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-12)

        spec   = np.abs(np.fft.rfft(mono[:args.blocksize] * window))
        energy = np.array([float(spec[idx].mean()) if idx.size else 0.0 for idx in band_idx])
        peaks     = np.maximum(energy, peaks * 0.995)             # slow AGC per band (for greens)
        loud_peak = max(rms, loud_peak * 0.995)                  # slow AGC for overall loudness
        bn = np.clip(energy / (peaks + 1e-9), 0.0, 1.0)          # per-band reactive level 0..1
        Lv = np.clip(rms / (loud_peak + 1e-9), 0.0, 1.0)         # overall loudness 0..1

        if rms < args.gate:                     # silence -> fade to dark, hold hue
            val_t, g0_t, g1_t = 0.0, 0.0, 0.0
            hue_t = hue
        else:
            val_t = Lv ** args.contrast         # brightness = how loud (bright only when loud)
            # hue = the true spectral BALANCE (not per-band-normalized), so bass->red,
            # mids->green, treble->blue actually reflect what's playing.
            prop = (energy * BAND_GAIN)
            prop = prop / (prop.sum() + 1e-9)
            w    = prop ** HUE_SHARP            # sharpen: the dominant band wins the color
            hue_t = float((BAND_HUE * w).sum() / (w.sum() + 1e-9))
            g0_t  = bn[0] ** 1.3                 # LD1: bass/kick
            g1_t  = bn[2] ** 1.2                 # LD2: treble sparkle

        # smoothing: hue drifts slowly; brightness has fast attack, slower release
        hue = hue * (1 - args.hue_smooth) + hue_t * args.hue_smooth
        val = val_t if val_t > val else val * 0.60 + val_t * 0.40
        g0  = g0_t  if g0_t  > g0  else g0  * 0.70 + g0_t  * 0.30
        g1  = g1_t  if g1_t  > g1  else g1  * 0.70 + g1_t  * 0.30

        r, g, b = colorsys.hsv_to_rgb(hue, args.sat, val)
        try:
            ser.write(bytes([SYNC, byte(g0), byte(g1), byte(r), byte(g), byte(b)]))
        except Exception:
            pass

        now = time.time()
        if now - last_print > 0.1:
            last_print = now
            colname = next(n for t, n in [(0.06, "red"), (0.13, "orange"), (0.19, "yellow"),
                                          (0.42, "green"), (0.55, "cyan"), (0.72, "blue"),
                                          (1.01, "violet")] if hue <= t)
            meter = "#" * int(val * 20)
            sys.stdout.write(f"\rhue={hue:0.2f}({colname:6s}) bright|{meter:<20}| ")
            sys.stdout.flush()

    try:
        with sd.InputStream(channels=1, samplerate=args.samplerate,
                            blocksize=args.blocksize, device=args.device,
                            dtype="float32", callback=callback):
            while True:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        try:
            ser.write(bytes([SYNC, 0, 0, 0, 0, 0]))
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
