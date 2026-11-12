# UART Protocol

Default UART settings:

- 115200 baud
- 8 data bits
- no parity
- 1 stop bit
- FPGA clock divisor: `round(12 MHz / 115200) = 104`

## LED Frame (retired)

Used by Phase 0 and Phase 1 only. The current bitstream no longer parses it: since
Phase 2C the FPGA picks the LED colors from the class it computes.

```text
0xFF ch0 ch1 ch2 ch3 ch4
```

Fields:

- `0xFF`: sync byte
- `ch0`: LD1 brightness, `0..254`
- `ch1`: LD2 brightness, `0..254`
- `ch2`: RGB red brightness, `0..254`
- `ch3`: RGB green brightness, `0..254`
- `ch4`: RGB blue brightness, `0..254`

Data bytes intentionally avoid `0xFF` so the FPGA can resynchronize.

## MFCC Feature Frame

Used by Phase 2.

```text
0xFE frame_index c0..c9 d0..d9 checksum
```

Fields:

- `0xFE`: MFCC sync byte
- `frame_index`: wraps at 255
- `c0..c9`: signed int8 MFCC coefficients encoded as unsigned bytes
- `d0..d9`: signed int8 first-order delta coefficients encoded as unsigned bytes
- `checksum`: low 8 bits of the sum of `frame_index` and the 20 coefficients

Each frame carries 20 coefficients. At 50 frames/s this is about
1150 bytes/s before UART overhead, still comfortably below 115200 baud.

## Class Reply Frame

Used by Phase 2C. Sent FPGA -> PC once per completed inference.

```text
0xCA class_id checksum
```

Fields:

- `0xCA`: class-reply sync byte
- `class_id`: predicted keyword class, `0..11` (index into `quant_params.json` classes)
- `checksum`: `(0xCA + class_id) & 0xFF`

The PC (`pc/kws_live.py --mode vectors`) streams one 49x20 feature window (49 MFCC
frames), then reads this reply and compares `class_id` against the Python fixed-point
reference (`fp_predicted_class`) to score the on-hardware gate.

## Signed Int8 Encoding

Python sends int8 coefficients as two's complement bytes:

```python
encoded = value & 0xFF
```

VHDL interprets them as `signed(7 downto 0)`.
