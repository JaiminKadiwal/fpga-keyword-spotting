# Fixed-Point Contract

This document defines the Python-to-VHDL numerical contract for Phase 2.

## Feature Tensor

- Shape: `49 x 10`
- Flattening: oldest frame first, coefficient order `c0..c9`
- Input type: signed int8
- Input scale: stored in `models/exported/quant_params.json`

## MLP Math

For each dense layer:

```text
acc[j] = bias[j] + sum_i input[i] * weight[j][i]
out[j] = clamp(round_shift(acc[j], shift), -128, 127)
```

Hidden layers apply ReLU after rescaling:

```text
out[j] = max(0, out[j])
```

Final layer does not need softmax in hardware. The FPGA outputs:

```text
class_id = argmax(logits)
```

## Accumulators

- Inputs: signed int8
- Weights: signed int8
- Bias: signed int32
- Accumulator: signed int32 minimum
- Output activation: signed int8

## Exported Artifacts

The Python export script owns the exact quantization parameters and emits:

- `weights_l1.mem`
- `bias_l1.mem`
- `weights_l2.mem`
- `bias_l2.mem`
- `weights_l3.mem`
- `bias_l3.mem`
- `quant_params.json`
- `golden_vectors.csv`

The VHDL implementation must be verified against `golden_vectors.csv` before
hardware testing.

`golden_vectors.csv` begins with `expected_class` and `python_predicted_class`,
followed by 490 signed int8 normalized feature values. Phase 2B should match
`python_predicted_class` bit-for-bit first, then compare model quality against
`expected_class`.
