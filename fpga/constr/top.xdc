## Cmod A7-35T constraints for the sound-reactive LED visualizer
## Pins taken verbatim from Digilent's Cmod-A7-Master.xdc

## 12 MHz system clock
set_property -dict { PACKAGE_PIN L17 IOSTANDARD LVCMOS33 } [get_ports { sysclk }];
create_clock -add -name sys_clk_pin -period 83.33 -waveform {0 41.66} [get_ports { sysclk }];

## USB-UART bridge
##   uart_txd_in  (J17) = data from the PC into the FPGA  -> our RX
##   uart_rxd_out (J18) = data from the FPGA to the PC    -> held idle-high
set_property -dict { PACKAGE_PIN J17 IOSTANDARD LVCMOS33 } [get_ports { uart_txd_in }];
set_property -dict { PACKAGE_PIN J18 IOSTANDARD LVCMOS33 } [get_ports { uart_rxd_out }];

## Green user LEDs (active high)
set_property -dict { PACKAGE_PIN A17 IOSTANDARD LVCMOS33 } [get_ports { led[0] }];
set_property -dict { PACKAGE_PIN C16 IOSTANDARD LVCMOS33 } [get_ports { led[1] }];

## RGB LED LD0 (active low / common anode)
set_property -dict { PACKAGE_PIN C17 IOSTANDARD LVCMOS33 } [get_ports { led0_r }];
set_property -dict { PACKAGE_PIN B16 IOSTANDARD LVCMOS33 } [get_ports { led0_g }];
set_property -dict { PACKAGE_PIN B17 IOSTANDARD LVCMOS33 } [get_ports { led0_b }];

## Buttons (active high)
set_property -dict { PACKAGE_PIN A18 IOSTANDARD LVCMOS33 } [get_ports { btn[0] }];
set_property -dict { PACKAGE_PIN B18 IOSTANDARD LVCMOS33 } [get_ports { btn[1] }];

## Configuration
set_property CFGBVS VCCO [current_design];
set_property CONFIG_VOLTAGE 3.3 [current_design];
