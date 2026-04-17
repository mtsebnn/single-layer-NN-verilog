// multiplier design 
// (embedded/fixed parameter because weight/bias values are fixed during inference) -> reduced hardware cost

module MUL_FIXED (
    input wire signed [7:0] a,    
    output wire signed [15:0] y  
);
    parameter WEIGHT = 0; // parameters default value
    wire signed [7:0] b = WEIGHT;

    assign y = a * b;
endmodule