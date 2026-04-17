`timescale 1ns/1ps
module network_testbench();
reg [7:0] inp [0:3];
reg clk;
wire signed [15:0] out [0:2];
reg [7:0] exp;
reg [7:0] res;

Network dut(
    .inp0(inp[0]),
    .inp1(inp[1]),
    .inp2(inp[2]),
    .inp3(inp[3]),
    .out0(out[0]),
    .out1(out[1]),
    .out2(out[2])
);

initial forever begin
	clk = 0; #5;
	clk = 1; #5;
end

// determine which class is actually classified
always @(out[0], out[1], out[2]) begin
    res = 8'd3;

    if (out[0] >= out[1] && out[0] >= out[2])
        res = 8'd0;
    else if (out[1] >= out[0] && out[1] >= out[2])
        res = 8'd1;
    else if (out[2] >= out[0] && out[2] >= out[1])
        res = 8'd2;
end

integer correct, wrong;

integer data_file;
integer status;
initial begin
    correct = 0;
    wrong = 0;
	inp[0] = 'd0;
    inp[1] = 'd0;
    inp[2] = 'd0;
    inp[3] = 'd0;
	#10;

	data_file = $fopen("neuronTV.dat", "r");
	while (!$feof(data_file)) begin
		status = $fscanf(data_file, "%d %d %d %d %d\n", inp[0], inp[1], inp[2], inp[3], exp);
		#10;
        if (exp === res)
            correct = correct + 1;
        else
            wrong = wrong + 1;
	end

    $display("\nClassified %d correctly and %d wrong\n", correct, wrong);
	$finish();
end
endmodule