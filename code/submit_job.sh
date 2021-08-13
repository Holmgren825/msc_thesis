#!/bin/bash

# Submit the job
for i in {0..74}; do sbatch --array=1-18 run_glacier_slurm.sh $i; done
