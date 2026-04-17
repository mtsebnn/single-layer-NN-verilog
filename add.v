// adder design
module ADD (
    input wire signed [15:0] a,
    input wire signed [15:0] b,
    output wire signed [15:0] y
);
    assign y = a + b;
endmodule