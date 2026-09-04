# Phase 2B behavioral simulation: replay Python fixed-point golden vectors
# through the kws_mlp datapath and check the predicted class matches.
#
#   vivado -mode batch -source fpga/vivado/sim_mlp.tcl
#
# Drives the standalone xsim tools (xvhdl/xelab/xsim). Runs inside build/sim so
# xsim byproducts stay out of the repo; absolute file paths are passed to the
# DUT/testbench via an xelab response file (avoids the .bat '=' mangling).
# Inputs:
#   models/exported/weights_l*.mem, bias_l*.mem
#   fpga/sim/vectors/golden_features.txt, golden_labels.txt
# Regenerate them if the model changed:
#   pc/export_weights.py --checkpoint models/checkpoints/kws_mlp.pt --cache <cache.npz>

set script_dir [file dirname [file normalize [info script]]]
set root [file normalize "$script_dir/../.."]
if {[info exists ::env(XILINX_VIVADO)]} {
    set bin [file normalize "$::env(XILINX_VIVADO)/bin"]
} else {
    # Fall back: strip bin/unwrapped/win64.o off the running executable path.
    set bin [file dirname [file dirname [file dirname [info nameofexecutable]]]]
}
set work "$root/build/sim/xsim_work"
set exp  "$root/models/exported"
set vec  "$root/fpga/sim/vectors"

file mkdir $work
cd $work

# xelab generics response file (one option per line; '=' preserved).
set gf [open "generics.f" w]
puts $gf "-generic_top \"FEAT_FILE=$vec/golden_features.txt\""
puts $gf "-generic_top \"LABEL_FILE=$vec/golden_labels.txt\""
puts $gf "-generic_top \"W1_FILE=$exp/weights_l1.mem\""
puts $gf "-generic_top \"B1_FILE=$exp/bias_l1.mem\""
puts $gf "-generic_top \"W2_FILE=$exp/weights_l2.mem\""
puts $gf "-generic_top \"B2_FILE=$exp/bias_l2.mem\""
puts $gf "-generic_top \"W3_FILE=$exp/weights_l3.mem\""
puts $gf "-generic_top \"B3_FILE=$exp/bias_l3.mem\""
close $gf

puts "=== xvhdl: compiling kws_mlp + testbench ==="
exec [file join $bin xvhdl.bat] "$root/fpga/src/kws_mlp.vhd" "$root/fpga/sim/tb_kws_mlp.vhd" >@ stdout 2>@ stdout

puts "=== xelab: building snapshot ==="
exec [file join $bin xelab.bat] tb_kws_mlp -s tb_snap -f generics.f >@ stdout 2>@ stdout

puts "=== xsim: running all golden vectors ==="
exec [file join $bin xsim.bat] tb_snap -R >@ stdout 2>@ stdout

puts "=== Phase 2B simulation finished. ==="
