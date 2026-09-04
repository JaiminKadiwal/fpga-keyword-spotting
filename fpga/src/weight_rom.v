// Synchronous ROM initialized from a hex .mem file via $readmemh.
//
// Vivado special-cases $readmemh and infers block RAM, so this works for large
// depths (e.g. 128*980 = 125,440 int8 weights). The equivalent VHDL textio
// initialization loop is unrolled at elaboration and trips the synthesis
// loop-iteration limit (65538), which is why the weight ROMs are Verilog.
//
// The read is registered (one-cycle latency) so the array maps to block RAM;
// kws_mlp's two-cycle MAC (S_MAC_SET then S_MAC_ACC) accounts for that latency.
module weight_rom #(
    parameter integer DEPTH     = 1024,
    parameter integer AWIDTH    = 10,
    parameter         INIT_FILE = ""
) (
    input  wire                 clk,
    input  wire [AWIDTH-1:0]    addr,
    output reg  signed [7:0]    dout
);
    (* rom_style = "block" *) reg signed [7:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(INIT_FILE, mem);
    end

    always @(posedge clk) begin
        dout <= mem[addr];
    end
endmodule
