# System Plan

## Goal

Build a keyword spotting system where the laptop supplies audio features and
the Cmod A7-35T performs neural network classification in FPGA fabric. The
visible output is a class-dependent LED color or pattern.

## Hardware/Software Split

The laptop handles:

- microphone capture at 16 kHz,
- windowing and MFCC extraction,
- optional laptop-side model execution during Phase 1,
- UART streaming of LED frames or MFCC feature frames.

The FPGA handles:

- UART receive and frame synchronization,
- feature ring buffering,
- fixed-point MLP inference,
- argmax, threshold/debounce, and LED output mapping.

This split keeps raw audio and MFCC DSP off the FPGA at first, while preserving
the important ECE/FPGA contribution: neural network inference in hardware.

## Data Flow

```text
Phase 1:
mic -> MFCC -> laptop model -> class/color -> 0xFF LED frame -> FPGA PWM LEDs

Phase 2:
mic -> MFCC -> 0xFE feature frame -> FPGA ring buffer -> FPGA MLP -> LEDs
```

## Model

Start with an MLP:

```text
Input: 49 x 10 int8 MFCC coefficients, flattened to 490
Layer 1: 490 -> 64, int8 weights, int32 accumulators, ReLU
Layer 2: 64 -> 32, int8 weights, int32 accumulators, ReLU
Layer 3: 32 -> 12, int8 weights, int32 accumulators
Output: argmax class
```

The class set is:

```text
0 silence
1 unknown
2 yes
3 no
4 up
5 down
6 left
7 right
8 on
9 off
10 stop
11 go
```

## Software Choices

Use Python for the main reproducible pipeline because it is GitHub-friendly and
works well with PyTorch/torchaudio/librosa. MATLAB with Signal Processing
Toolbox is useful for validating MFCC math and making course-quality plots.

Stretch tools:

- hls4ml: useful if trying a Keras/TensorFlow DS-CNN through HLS.
- FINN + Brevitas: useful for a research-style quantized FPGA flow.

## Growth Path

Do not replace the MLP with a DS-CNN until the MLP passes hardware inference.
The DS-CNN path should be treated as a Phase 3 extension, not the main path.
