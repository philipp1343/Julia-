# Demo experiment

Example on how to run an experiment on das-5. All command below are to be run on the LOGIN node of das5.

Disclaimer: This is just a simple example. Elaborate more in your experiments.

1. Clone this repo to das5
2. Launch the configure script `$ ./step_01_config.sh`
3. Launch a job `$ sbatch step_02_launch_job.sh`
4. Modify the parameters `parts_per_dir` and `nodes_per_dir` in  `step_02_launch_job.sh` and launch it again.
5. Repeat this several times to collect several result files
6. Analyze the results with `step_03_analysis.jl`. This is a Julia script: open julia and include it in the REPL with `inlcude("step_03_analysis.jl")`.
