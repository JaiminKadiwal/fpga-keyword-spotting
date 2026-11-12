# Fixed-Point Contract

The Python-to-VHDL numerical contract that `pc/export_weights.py` and
`fpga/src/kws_mlp.vhd` both follow.

## Feature Tensor

- Shape: `49 x 20`
- Coefficient order per frame: `c0..c9` MFCC, then `d0..d9` first-order deltas
- Flattening: oldest frame first, then coefficient order above (49 x 20 = 980 total)
- Front-end normalization (laptop side, identical for train/live): per-window peak
  normalization before MFCC, then per-window CMVN (cepstral mean/variance) over the 49
  frames. The FPGA therefore receives already-normalized features.
- Input type: signed int8
- Input scale: stored in `models/exported/quant_params.json`

## FPGA Resource Budget (xc7a35t)

- int8 weight bytes: `980*128 + 128*64 + 64*12 = 134,400` (~131 KB)
- Available BRAM: 50 x 36 Kb = 1,800 Kb (~225 KB) -> weights use ~58%. The built
  design uses 71% (35.5 of 50 tiles) including biases and feature memory. A 256-wide first hidden layer would exceed BRAM, so
  the hidden width is capped at 128.

## MLP Math (exact integer pipeline)

`pc/export_weights.py` is the reference implementation; the VHDL in
`fpga/src/kws_mlp.vhd` reproduces it bit-for-bit. Each dense layer computes an
int32 accumulator:

```text
acc[j] = bias[j] + sum_i input[i] * weight[j][i]
```

Hidden layers (l1, l2) apply ReLU, then a round-half-up arithmetic right shift by
the per-layer `shift` from `quant_params.json`, then clamp to int8 [0, 127]:

```text
v      = max(0, acc[j])
out[j] = clamp( (v + (1 << (shift-1))) >> shift, 0, 127 )
```

The output layer (l3) applies no shift and no ReLU. The FPGA reports the argmax
of the int32 logits, breaking ties toward the lowest index (matches numpy):

```text
class_id = argmax(logits)
```

## Quantization scales

- Weights: symmetric int8, `weight_scale = max_abs(W) / 127`.
- Input features: symmetric int8, `input_scale` chosen from the calibration set.
- Bias: `bias[j] = round(bias_float[j] / (s_in * weight_scale))`, where `s_in` is
  the cumulative input scale of that layer (`input_scale` for l1, then
  `s_in * weight_scale * 2^shift` for the next layer).
- `shift` per hidden layer is calibrated so post-ReLU activations fit int8.

## Accumulators

- Inputs: signed int8
- Weights: signed int8
- Bias: signed int32
- Accumulator: signed int32
- Hidden output activation: signed int8 [0, 127]
- Output logits: signed int32 (argmax only)

## Exported Artifacts

The Python export script owns the exact quantization parameters and emits:

- `weights_l1.mem`
- `bias_l1.mem`
- `weights_l2.mem`
- `bias_l2.mem`
- `weights_l3.mem`
- `bias_l3.mem`
- `quant_params.json`
- `golden_vectors.csv`, `golden_vectors.json`

The VHDL implementation must be verified against `golden_vectors.csv` before
hardware testing.

`golden_vectors.csv` columns: `expected_class`, `fp_predicted_class` (the integer
reference), `float_predicted_class`, then 980 signed int8 features. The VHDL must
match `fp_predicted_class` on every row; `expected_class` measures model quality.
