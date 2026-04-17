// hardware implementation single-layer neural network
// could also build an external neuron module and use it to simplify this network module

module Network (
    input wire signed [7:0] inp0,
    input wire signed [7:0] inp1,
    input wire signed [7:0] inp2,
    input wire signed [7:0] inp3,
    output wire signed [15:0] out0,
    output wire signed [15:0] out1,
    output wire signed [15:0] out2
    // Single layer network with 4 inputs and 3 outputs
    // -> weight matrix shape: [3 outputs x 4 inputs]
);
    // define neurons
    parameter signed [7:0] w00 = -8'sd127;
    parameter signed [7:0] w01 = 8'sd59;
    parameter signed [7:0] w02 = -8'sd36;
    parameter signed [7:0] w03 = -8'sd10;
    parameter signed [15:0] b0 = 16'sd5231;

    parameter signed [7:0] w10 = 8'sd93;
    parameter signed [7:0] w11 = 8'sd23;
    parameter signed [7:0] w12 = -8'sd23;
    parameter signed [7:0] w13 = -8'sd69;
    parameter signed [15:0] b1 = -16'sd2028;

    parameter signed [7:0] w20 = 8'sd5;
    parameter signed [7:0] w21 = -8'sd100;
    parameter signed [7:0] w22 = 8'sd58;
    parameter signed [7:0] w23 = 8'sd61;
    parameter signed [15:0] b2 = -16'sd2939;

    // weight input multiplication
    wire signed [15:0] m00_out;
    wire signed [15:0] m01_out;
    wire signed [15:0] m02_out;
    wire signed [15:0] m03_out;
    MUL_FIXED #(.WEIGHT(w00)) mul00 (.a(inp0), .y(m00_out));
    MUL_FIXED #(.WEIGHT(w01)) mul01 (.a(inp1), .y(m01_out));
    MUL_FIXED #(.WEIGHT(w02)) mul02 (.a(inp2), .y(m02_out));
    MUL_FIXED #(.WEIGHT(w03)) mul03 (.a(inp3), .y(m03_out));

    wire signed [15:0] m10_out;
    wire signed [15:0] m11_out;
    wire signed [15:0] m12_out;
    wire signed [15:0] m13_out;
    MUL_FIXED #(.WEIGHT(w10)) mul10 (.a(inp0), .y(m10_out));
    MUL_FIXED #(.WEIGHT(w11)) mul11 (.a(inp1), .y(m11_out));
    MUL_FIXED #(.WEIGHT(w12)) mul12 (.a(inp2), .y(m12_out));
    MUL_FIXED #(.WEIGHT(w13)) mul13 (.a(inp3), .y(m13_out));

    wire signed [15:0] m20_out;
    wire signed [15:0] m21_out;
    wire signed [15:0] m22_out;
    wire signed [15:0] m23_out;
    MUL_FIXED #(.WEIGHT(w20)) mul20 (.a(inp0), .y(m20_out));
    MUL_FIXED #(.WEIGHT(w21)) mul21 (.a(inp1), .y(m21_out));
    MUL_FIXED #(.WEIGHT(w22)) mul22 (.a(inp2), .y(m22_out));
    MUL_FIXED #(.WEIGHT(w23)) mul23 (.a(inp3), .y(m23_out));

    // sum products
    wire signed [15:0] add00_out;
    wire signed [15:0] add01_out;
    wire signed [15:0] add02_out;
    wire signed [15:0] bias0_ext;
    assign bias0_ext = b0;
    ADD add00(.a(m00_out), .b(m01_out), .y(add00_out));
    ADD add01(.a(m02_out), .b(m03_out), .y(add01_out));
    ADD add02(.a(add00_out), .b(add01_out), .y(add02_out));
    ADD add03(.a(add02_out), .b(bias0_ext), .y(out0));

    wire signed [15:0] add10_out;
    wire signed [15:0] add11_out;
    wire signed [15:0] add12_out;
    wire signed [15:0] bias1_ext;
    assign bias1_ext = b1;
    ADD add10(.a(m10_out), .b(m11_out), .y(add10_out));
    ADD add11(.a(m12_out), .b(m13_out), .y(add11_out));
    ADD add12(.a(add10_out), .b(add11_out), .y(add12_out));
    ADD add13(.a(add12_out), .b(bias1_ext), .y(out1));

    wire signed [15:0] add20_out;
    wire signed [15:0] add21_out;
    wire signed [15:0] add22_out;
    wire signed [15:0] bias2_ext;
    assign bias2_ext = b2;
    ADD add20(.a(m20_out), .b(m21_out), .y(add20_out));
    ADD add21(.a(m22_out), .b(m23_out), .y(add21_out));
    ADD add22(.a(add20_out), .b(add21_out), .y(add22_out));
    ADD add23(.a(add22_out), .b(bias2_ext), .y(out2));
endmodule