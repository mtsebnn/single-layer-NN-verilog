# only for linux/windows simulation, with modelsim/questa run: vsim -do simulation.tcl

# on mac run: iverilog -o sim network_testbench.v network.v add.v mul.v
#             vvp sim

vlib work
vlog testbench_network.v
vlog network.v
vlog add.v
vlog mul_fixed.v
vsim work.testbench_network
run 10 us
quit -f