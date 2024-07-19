# The convergence-adaptive enhanced sampling method
The convergence-adaptive roundtrip enhanced sampling method enables rapid and accurate FEP calculations.

## Procedures to run simulaitons of a thermodynamic process with the CAR method are as follow:
1. Step 1: Prepare the input files.
    - coordinate file and topology file;
    - a json file specifies the lambda scheme; e.x. **input_temp/lambdas.json**
    - a json file includes input settings for AMBER; e.x. **input_temp/input*.json**. Note:  __SCMASK__ should be modified to unique atoms list for each pair as input.json shown in directory of each pair.
    - settings of CAR method; e.x. **input_temp/car_run_input.txt**. Note: Some settings need to be changed according to the working directory path and filenames.
2. Step 2: Start the CAR enhanced adaptive sampling program.
    - `python $CAR_SCRIPT_DIR/car_converge_control.py -i car_run_input.txt`

## Details of car_run_input.txt
```
## input file for convergence adaptive control md run
[normal_alc_md]
simulation_software     = amber # the simulation software
coordinate_file         = __COOR__ # the initial input coordinate_file, can be '.prmcrd' or '.rst7'
topology_file           = __TOPO__ # the initial input topology_file
prod_md_time            = 10 # simulation time of each state in unit of ps
mbar_lambda_dict_file   = lambdas.json # the lambda schemes
input_file              = input.json # settings for simulations

[segmented_md_control]
segment_lambda_step     = 0.2 # determine how to divide the thermodynamic process into parts
num_neighbors_state     = 5 # the number of neighboring states to record potential energies when simulating specified state.
min_reitera_times       = 2 # specify the minimun reiteration times (an iteration includes a switchback round and a restart forward round) of a part
max_reitera_times       = 50 # specify the maximum reiteration times (an iteration includes a switchback round and a restart forward round) to terminate simulations of a part
error_max_lambda_0to1   = 0.15 # specify the uncertainty allowed for an entire thermodynamic process
ifrun_preliminary_md    = True # whether to run pre-equilibrium MD at initial state 0
ifuse_initial_rst       = True # whether to start MD with initial input coordinate_file, else specify a file_path to run continuous MD
ifoverwrite             = True # if True, history directories will be removed, else concat the results of new simulation to existed files.
```
