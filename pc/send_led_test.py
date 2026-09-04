#!/usr/bin/env python3
"""
Link/LED self-test - no microphone needed.

Sweeps each channel so you can confirm the UART path and every LED:
  1) LD1 fades up/down, 2) LD2, 3) RGB red, 4) green, 5) blue,
  6) a slow rainbow on the RGB with both greens pulsing.

Usage: python send_test.py            (auto-detect COM port)
       python send_test.py --port COM8
"""
import argparse, math, time, sys
import serial
import serial.tools.list_ports as list_ports

SYNC = 0xFF
FTDI_VIDPID = (0x0403, 0x6010)


def find_port():
    for p in list_ports.comports():
        if (p.vid, p.pid) == FTDI_VIDPID:
            return p.device
    for p in list_ports.comports():
        if "USB Serial" in (p.description or ""):
            return p.device
    return None


def send(ser, ch):
    ser.write(bytes([SYNC] + [max(0, min(254, int(v))) for v in ch]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()
    port = args.port or find_port()
    if not port:
        print("No Cmod A7 serial port found (try --port COMxx).")
        sys.exit(1)
    ser = serial.Serial(port, args.baud, timeout=0)
    print(f"Testing on {port}. Watch the LEDs. Ctrl+C to stop.")
    names = ["LD1 (green)", "LD2 (green)", "RGB red", "RGB green", "RGB blue"]
    try:
        for i, name in enumerate(names):          # fade each channel in turn
            print(f"  -> {name}")
            for t in range(0, 100):
                v = int((math.sin(t / 100 * math.pi)) * 254)
                ch = [0, 0, 0, 0, 0]
                ch[i] = v
                send(ser, ch)
                time.sleep(0.01)
        print("  -> rainbow sweep (RGB) + green pulse")
        t = 0
        while True:                               # cool finale until Ctrl+C
            r = int((math.sin(t) * 0.5 + 0.5) * 254)
            g = int((math.sin(t + 2.09) * 0.5 + 0.5) * 254)
            b = int((math.sin(t + 4.19) * 0.5 + 0.5) * 254)
            p = int((math.sin(t * 2) * 0.5 + 0.5) * 254)
            send(ser, [p, 254 - p, r, g, b])
            time.sleep(0.02)
            t += 0.05
    except KeyboardInterrupt:
        send(ser, [0, 0, 0, 0, 0])
        ser.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
