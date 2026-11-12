# Verification

Each phase had a gate that had to pass before the next one started. The commit for each phase records the
command and the result.

| Phase | Gate | Command | Result |
|---|---|---|---|
| 0 | Baseline UART LED design builds, programs, and drives all LEDs | `vivado -mode batch -source fpga/vivado/build.tcl`, then `program.tcl` | Pass: LD1, LD2 and RGB channels responded |
| 1 | Laptop KWS drives LED colors from prerecorded WAVs and live mic | `python pc/kws_live.py --mode wav-dir ...`, then `--mode live` | Pass: 92 to 100% on sampled target classes; stable live predictions |
| 2A | Quantized model reaches 80% validation accuracy | `python pc/train_mlp.py ...`, then `python pc/export_weights.py ...` | Pass: val 81.6%, test 80.1%; fixed point val 81.3%, 98.3% agreement with float |
| 2B | VHDL MLP matches the Python fixed-point reference on every golden vector | `.\run.ps1 -Sim` | Pass: 128/128 bit exact in xsim |
| 2C | FPGA matches the golden vectors over UART | `python pc/kws_live.py --mode vectors --vectors models/exported/golden_vectors.csv` | Pass: 128/128; 71% BRAM, timing met (WNS +68 ns at 12 MHz) |
| 2D | Live mic features drive on-FPGA inference | `.\run.ps1` | Pass: live quantizer reproduces all 128 golden vectors byte for byte; keywords light the expected colors |

## Phase 2A training command

```powershell
python pc/train_mlp.py --epochs 80 --max-per-class 4000 --dropout 0.15 --no-balance --cache data/mfcc_cache_aug2.npz --rebuild-cache --out models/checkpoints/kws_mlp.pt
python pc/export_weights.py --checkpoint models/checkpoints/kws_mlp.pt --cache data/mfcc_cache_aug2.npz
```

The export writes the int8 `.mem` weights and biases, `quant_params.json`, `golden_vectors.csv`, and the
simulation vectors under `fpga/sim/vectors/`.

## Microphone check before live runs

```powershell
python pc/mic_check.py --list
python pc/mic_check.py --device 1 --samplerate 44100 --channels 1 --seconds 10
```

The RMS and peak bars should jump when speaking. If they stay flat, live predictions are not meaningful.
