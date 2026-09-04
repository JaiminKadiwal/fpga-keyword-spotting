# UART Protocol

Default UART settings:

- 115200 baud
- 8 data bits
- no parity
- 1 stop bit
- FPGA clock divisor: `round(12 MHz / 115200) = 104`

## LED Frame

Used by Phase 0 and Phase 1.

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
0xFE frame_index c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 checksum
```

Fields:

- `0xFE`: MFCC sync byte
- `frame_index`: wraps at 255
- `c0..c9`: signed int8 MFCC coefficients encoded as unsigned bytes
- `checksum`: low 8 bits of the sum of `frame_index` and `c0..c9`

At 50 frames/s this is about 650 bytes/s before UART overhead, which is still
comfortably below 115200 baud.

## Signed Int8 Encoding

Python sends int8 coefficients as two's complement bytes:

```python
encoded = value & 0xFF
```

VHDL interprets them as `signed(7 downto 0)`.
