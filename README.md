# fpga-keyword-spotting

Live speech keyword spotting with the neural network running on an FPGA. A laptop turns microphone audio into
MFCC features and streams them over UART to a Digilent Cmod A7-35T (Artix-7). An int8 MLP written in VHDL
classifies 12 classes on chip and drives the LED color. The VHDL matches the Python fixed-point model bit for bit.

## Results

| Check | Result |
|---|---|
| Model accuracy (Google Speech Commands v2, 12 classes) | 80.1% test, 81.6% validation (float); fixed point 81.3% validation, 98.3% agreement with float |
| VHDL vs Python fixed-point reference, xsim | 128/128 golden vectors, same class |
| FPGA vs Python, golden vectors streamed over UART | 128/128 |
| Live mic, FPGA inference | keywords light the expected LED colors |
| Resources (xc7a35t) | 600 LUTs (2.9%), 307 flip-flops, 35.5 of 50 BRAM tiles (71%), 1 DSP |
| Timing at 12 MHz | met, WNS +67.8 ns |
| Inference latency | about 2.7e5 cycles per window, about 22 ms at 12 MHz (measured in xsim) |

How each phase was gated: [docs/verification_plan.md](docs/verification_plan.md).

## How it works

```
Laptop                                              Cmod A7-35T
------                                              -----------
mic 44.1 kHz -> 16 kHz, peak normalize              UART RX, 115200 8N1
MFCC (10) + delta (10) per frame, 49 frames         frame parser: 0xFE, index, 20 coeffs, checksum
per-window CMVN, quantize to int8      --UART-->    kws_mlp: 980 -> 128 -> 64 -> 12, int8 weights in BRAM
                                       <--UART--    argmax -> class reply (0xCA, class, checksum)
                                                    class_to_led -> PWM on RGB LED and LD1/LD2
```

- **Fixed-point contract.** int8 activations and weights, int32 bias and accumulator. Hidden layers apply ReLU,
  then a round-half-up arithmetic right shift calibrated per layer (9 and 8), then clamp to int8. The output layer
  is an argmax over int32 logits. Details: [docs/fixed_point_contract.md](docs/fixed_point_contract.md).
- **Datapath.** One multiply-accumulate in a two-cycle set/accumulate step that covers the block-RAM read
  latency. The 125,440 layer-1 weights load through a small Verilog `$readmemh` ROM, because a VHDL textio init
  loop of that size hit Vivado's synthesis loop limit.
- **Protocol.** Feature frames carry a counter and checksum because int8 coefficients can equal any byte value,
  including the sync byte. Details: [docs/uart_protocol.md](docs/uart_protocol.md).
- **Why split it this way.** The board has no audio input and MFCC is DSP heavy, so the laptop does the front end
  and the FPGA does the part that matters here: neural network inference in fabric.

## Quick start

Needs Windows, Vivado 2026.1 (the free tier covers the xc7a35t) and Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

.\run.ps1 -Sim      # xsim: VHDL vs Python on 128 golden vectors, no board needed
.\run.ps1           # with the board plugged in: program over JTAG and start the live demo
```

`run.ps1` options: `-Build` rebuilds the bitstream first, `-JustRun` skips programming, `-Port COM8` and
`-Device 1` pick the serial port and microphone. Speak `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`,
`stop` or `go`; each printed line is the FPGA's own decision.

Stream the golden vectors to the board and compare every reply:

```powershell
python pc/kws_live.py --mode vectors --vectors models/exported/golden_vectors.csv
```

## Retrain and re-export

```powershell
python pc/prepare_speech_commands.py      # downloads Speech Commands v2 into data/speech_commands/
python pc/train_mlp.py --epochs 80 --max-per-class 4000 --dropout 0.15 --no-balance --cache data/mfcc_cache_aug2.npz --rebuild-cache --out models/checkpoints/kws_mlp.pt
python pc/export_weights.py --checkpoint models/checkpoints/kws_mlp.pt --cache data/mfcc_cache_aug2.npz
```

The export writes the `.mem` weight and bias files, `quant_params.json` and the golden vectors that the
simulation and the board check against. Datasets and checkpoints stay out of git.

## Layout

```
fpga/src/      top.vhd, kws_mlp.vhd, uart_rx/tx.vhd, class_to_led.vhd, pwm8.vhd, weight_rom.v
fpga/sim/      tb_kws_mlp.vhd and golden vectors
fpga/constr/   Cmod A7 pin constraints
fpga/vivado/   build.tcl, program.tcl, sim_mlp.tcl (batch mode)
pc/            MFCC front end, training, export, live and vector streaming, mic check
models/        exported int8 weights, biases, quantization parameters, golden vectors
docs/          fixed-point contract, UART protocol, verification record
```

## Credits

Trained on [Google Speech Commands v2](https://arxiv.org/abs/1804.03209) (Warden, 2018), licensed CC BY 4.0.

## License

MIT
