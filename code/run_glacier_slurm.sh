#!/bin/bash
#
#SBATCH --job-name=peak_water
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --mail-user=erik.holmgren@student.uibk.ac.at
#SBATCH --mail-type=ALL
#SBATCH --qos=low
#SBATCH --output=/home/users/eholmgren/work/logs/slurm_%j.out

# We should abort if a single step fails.
set -e

# Set up where to store the data. Slurm makes sure that the directory
# /work/username exist and is writeable by the job user.
OGGM_WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/wd"
mkdir -p "$OGGM_WORKDIR"
export OGGM_WORKDIR
echo "Workdir for this run: $OGGM_WORKDIR"

# Some more configs for OGGM.
export OGGM_EXTRACT_DIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/oggm_tmp"
export OGGM_USE_MULTIPROCESSING=1

# Useful defaults
export LRU_MAXSIZE=1000

# Output directory
OGGM_OUTDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/out"
export OGGM_OUTDIR
echo "Output dir for this run: $OGGM_OUTDIR"

# All commands in the EOF block run inside of the container
# Adjust container version to your needs, they are guaranteed to never change after their respective day has passed.
srun -n 1 -c "${SLURM_JOB_CPUS_PER_NODE}" singularity exec /home/users/eholmgren/images/oggm_20210810.sif bash -s <<EOF
  set -e
  # Setup a fake home dir inside of our workdir, so we don't clutter the actual shared homedir with potentially incompatible stuff.
  export HOME="$OGGM_WORKDIR/fake_home"
  mkdir "\$HOME"
  # Create a venv that _does_ use system-site-packages, since everything is already installed on the container.
  # We cannot work on the container itself, as the base system is immutable.
  python3 -m venv --system-site-packages "$OGGM_WORKDIR/oggm_env"
  source "$OGGM_WORKDIR/oggm_env/bin/activate"
  # Make sure latest pip is installed
  pip install --upgrade pip setuptools
  # OPTIONAL: install OGGM latest
  pip install --no-deps "git+https://github.com/OGGM/oggm.git@v1.5.0"
  pip install pathos
  # Increase number of allowed open file descriptors
  ulimit -n 65000
  # Finally, the run
  python run_gha.py "$1"
EOF

# Write out
echo "Copying files..."
rsync -avzh "$OGGM_OUTDIR/" /home/users/eholmgren/work/gha_basins
