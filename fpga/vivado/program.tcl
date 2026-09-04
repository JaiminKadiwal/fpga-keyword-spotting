# Program the Cmod A7 with the repo-local KWS/LED bitstream.
set script_dir [file dirname [file normalize [info script]]]
set root [file normalize "$script_dir/../.."]
set bit "$root/build/vivado/kws_fpga_led.runs/impl_1/top.bit"

if {![file exists $bit]} {
    puts "=== FATAL: bitstream missing: $bit ==="
    exit 1
}

open_hw_manager
connect_hw_server -allow_non_jtag
set targets [get_hw_targets]
if {[llength $targets] == 0} { puts "=== FATAL: no JTAG target (board unplugged?) ==="; exit 2 }
open_hw_target [lindex $targets 0]
set dev [lindex [get_hw_devices] 0]
current_hw_device $dev
refresh_hw_device -quiet $dev
puts "=== DEVICE: $dev  PART: [get_property PART $dev] ==="

set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
refresh_hw_device $dev
catch { puts "=== DONE_PIN (1=configured): [get_property REGISTER.CONFIG_STATUS.BIT14_DONE_PIN $dev] ===" }
close_hw_target
disconnect_hw_server
close_hw_manager
puts "=== PROGRAM COMPLETE ==="
exit 0
