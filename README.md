# FPGA Keyword Spotting LED Demo

JHU ECE project: spoken keywords control LEDs on a Digilent Cmod A7-35T. The
laptop handles audio capture and MFCC feature extraction; the FPGA receives
compact feature frames over USB-UART and performs neural network inference.

## Architecture

```text
Laptop microphone
  -> Python or MATLAB audio front end
  -> MFCC feature extraction
  -> UART feature stream
  -> Cmod A7-35T FPGA
  -> fixed-point neural network inference
  -> LED color output
```

The repository begins with a preserved baseline from the existing audio LED
visualizer:

```text
Laptop audio features -> UART 115200 -> FPGA frame parser -> PWM LEDs
```

That baseline is Phase 0. Later phases replace the laptop-side music visualizer
with keyword spotting and then move the neural network classifier into FPGA
fabric.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `python` launches the GTKWave/Icarus Python from `C:\iverilog\gtkwave\bin`,
install Python 3.11+ from python.org or Microsoft Store and put it before
GTKWave on `PATH`.

Build and program the baseline FPGA design:

```powershell
vivado -mode batch -source fpga/vivado/build.tcl
vivado -mode batch -source fpga/vivado/program.tcl
python pc/send_led_test.py
```

Run the existing sound-reactive LED baseline:

```powershell
python pc/audio_leds.py
```

Capture one MFCC window from the microphone:

```powershell
python pc/capture_mfcc.py --out fpga/sim/vectors/sample_mfcc.csv
```

## Phase Plan

- Phase 0: preserve and verify the existing UART LED design.
- Phase 1: laptop runs KWS and sends keyword colors to the FPGA.
- Phase 2A: train and quantize a small MLP on Speech Commands.
- Phase 2B: simulate the VHDL MLP against Python golden vectors.
- Phase 2C: run FPGA inference with prerecorded feature vectors.
- Phase 2D: run live MFCC streaming into FPGA inference.
- Phase 3: optional DS-CNN with hls4ml or FINN.

Every phase has a required gate in `docs/verification_plan.md`. Record results
in the commit message before moving forward.

## Repository Hygiene

Vivado generated state, datasets, bitstreams, and large model checkpoints are
ignored. Git should contain source code, documentation, scripts, and lightweight
fixed-point export artifacts only.

## Board

- Digilent Cmod A7-35T
- FPGA part: `xc7a35tcpg236-1`
- Clock: 12 MHz
- UART: FTDI USB-UART, default 115200 baud
- Outputs: LD1, LD2, RGB LED LD0
