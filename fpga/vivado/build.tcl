# Build the baseline KWS/LED Vivado project from repo-relative source paths.
set script_dir [file dirname [file normalize [info script]]]
set root [file normalize "$script_dir/../.."]
set part "xc7a35tcpg236-1"
set build_dir "$root/build/vivado"

file mkdir $build_dir
create_project kws_fpga_led $build_dir -part $part -force
set_property target_language VHDL [current_project]

add_files [list \
    "$root/fpga/src/uart_rx.vhd" \
    "$root/fpga/src/uart_tx.vhd" \
    "$root/fpga/src/pwm8.vhd" \
    "$root/fpga/src/class_to_led.vhd" \
    "$root/fpga/src/mfcc_ring_buffer.vhd" \
    "$root/fpga/src/weight_rom.v" \
    "$root/fpga/src/kws_mlp.vhd" \
    "$root/fpga/src/top.vhd" ]
add_files -fileset constrs_1 "$root/fpga/constr/top.xdc"

# Weight/bias ROMs read by kws_mlp via textio at synthesis elaboration. Add the
# files to the project and pass absolute (normalized, forward-slash) paths as
# top-level generics so textio resolves them regardless of the run directory.
set exp "$root/models/exported"
add_files [list \
    "$exp/weights_l1.mem" "$exp/bias_l1.mem" \
    "$exp/weights_l2.mem" "$exp/bias_l2.mem" \
    "$exp/weights_l3.mem" "$exp/bias_l3.mem" ]

set_property top top [current_fileset]
set_property generic [list \
    "W1_FILE=$exp/weights_l1.mem" "B1_FILE=$exp/bias_l1.mem" \
    "W2_FILE=$exp/weights_l2.mem" "B2_FILE=$exp/bias_l2.mem" \
    "W3_FILE=$exp/weights_l3.mem" "B3_FILE=$exp/bias_l3.mem" ] [current_fileset]
update_compile_order -fileset sources_1
puts "=== PROJECT CREATED: [current_project] part [get_property PART [current_project]] top=[get_property top [current_fileset]] ==="

launch_runs synth_1 -jobs 8
wait_on_run synth_1
puts "=== SYNTH: [get_property STATUS [get_runs synth_1]] progress=[get_property PROGRESS [get_runs synth_1]] ==="
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { puts "=== FATAL: synth failed ==="; exit 1 }

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
puts "=== IMPL: [get_property STATUS [get_runs impl_1]] progress=[get_property PROGRESS [get_runs impl_1]] ==="
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { puts "=== FATAL: impl/bitstream failed ==="; exit 1 }

set bit "$build_dir/kws_fpga_led.runs/impl_1/top.bit"
if {[file exists $bit]} {
    puts "=== BITSTREAM_OK: $bit ([file size $bit] bytes) ==="
} else {
    puts "=== FATAL: bitstream missing ==="; exit 1
}
puts "=== BUILD COMPLETE ==="
exit 0
