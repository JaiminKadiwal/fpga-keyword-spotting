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

On this machine the discovered full paths are:

```powershell
& 'C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat' -mode batch -source fpga/vivado/build.tcl
& 'C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat' -mode batch -source fpga/vivado/program.tcl
python pc/send_led_test.py --port COM8
```

Run the existing sound-reactive LED baseline:

```powershell
python pc/audio_leds.py
```

Capture one MFCC window from the microphone:

```powershell
python pc/capture_mfcc.py --out fpga/sim/vectors/sample_mfcc.csv
```

Prepare the Speech Commands dataset when ready for real training:

```powershell
python pc/prepare_speech_commands.py
python pc/train_mlp.py --epochs 20 --max-per-class 1000
python pc/export_weights.py
```

Quick software smoke test without downloading the dataset:

```powershell
python pc/train_mlp.py --smoke --epochs 3 --out build/smoke/kws_mlp.pt
python pc/export_weights.py --checkpoint build/smoke/kws_mlp.pt --out build/smoke/exported
```

Before live KWS, verify the microphone path:

```powershell
python pc/mic_check.py --list
python pc/mic_check.py --probe
python pc/mic_check.py --device 23 --samplerate 48000 --seconds 8
python pc/mic_check.py --device 15 --samplerate 48000 --channels 4 --seconds 8
```

The RMS/peak bars should jump when speaking. If they stay flat, do not trust
live KWS predictions yet.

Run a prerecorded WAV folder gate before trusting live audio:

```powershell
python pc/kws_live.py --mode wav-dir --wav-dir data/local_test_wavs --limit 30
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
