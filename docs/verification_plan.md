# Verification Plan

Do not advance phases without recording the command, observed result, pass/fail
state, and next action in the commit message.

## Phase 0 Gate: Baseline LED Link

Commands:

```powershell
vivado -mode batch -source fpga/vivado/build.tcl
vivado -mode batch -source fpga/vivado/program.tcl
python pc/send_led_test.py
```

Known local full-path equivalents:

```powershell
& 'C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat' -mode batch -source fpga/vivado/build.tcl
& 'C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat' -mode batch -source fpga/vivado/program.tcl
python pc/send_led_test.py --port COM8
```

Pass condition:

- bitstream builds,
- board programs,
- LD1, LD2, red, green, and blue channels visibly respond.

## Phase 1 Gate: Laptop KWS To LEDs

Commands:

```powershell
python pc/mic_check.py --probe
python pc/mic_check.py --device 23 --samplerate 48000 --seconds 8
python pc/mic_check.py --device 15 --samplerate 48000 --channels 4 --seconds 8
python pc/kws_live.py --model models/checkpoints/kws_mlp.pt --mode wav-dir --wav-dir data/local_test_wavs
python pc/kws_live.py --model models/checkpoints/kws_mlp.pt --mode live
```

Pass condition:

- microphone RMS/peak visibly jumps when speaking,
- at least 80% correct on a small local WAV test set,
- silence does not trigger a keyword color,
- live mic can reliably trigger `yes`, `no`, and `stop`.

## Phase 2A Gate: Quantized Python Model

Commands:

```powershell
python pc/train_mlp.py --data data/speech_commands --epochs 20 --max-per-class 1000
python pc/export_weights.py --checkpoint models/checkpoints/kws_mlp.pt --out models/exported
```

Pass condition:

- quantized model reaches at least 75-80% validation accuracy,
- exported golden vectors are generated.

## Phase 2B Gate: VHDL Simulation

Command:

```powershell
vivado -mode batch -source fpga/vivado/sim_mlp.tcl
```

Pass condition:

- VHDL argmax class matches Python golden output for every test vector.

## Phase 2C Gate: Hardware Inference From Prerecorded Features

Command:

```powershell
python pc/kws_live.py --mode vectors --vectors models/exported/golden_vectors.csv
```

Pass condition:

- no UART framing loss during a 5-minute stream,
- FPGA LED/class output agrees with expected vector labels.

## Phase 2D Gate: Live FPGA Inference

Command:

```powershell
python pc/kws_live.py --mode live --fpga-inference
```

Pass condition:

- target words produce stable expected LED colors,
- silence stays idle/off,
- repeated triggers are debounced.
