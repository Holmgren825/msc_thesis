#!/bin/bash

# List of basins
export GHA_BASINS="6255"

# Submit the job
for i in $GHA_BAINS; do sbatch --array=1-18 run_glacier_slurm.sh $i; done
