#!/bin/bash
#SBATCH --job-name=phoscrosstalk_mpi
#SBATCH --partition=BigMem
#SBATCH --nodes=3                 # max possible here
#SBATCH --ntasks-per-node=64       # 24 MPI ranks total
#SBATCH --cpus-per-task=1
#SBATCH --time=10-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# 1) Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phoscrosstalk

# 2) Project dir
#cd /home/abhinav/Documents/phoscrosstalk
mkdir -p logs

# 3) Setup Environment for MPI
# Force single threading per MPI task so they don't fight for resources
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 4) Run with mpiexec (FIXED)
# We use 'mpiexec' from the conda environment instead of 'srun'.
# This ensures the launcher matches the installed library.
# We pass -n $SLURM_NTASKS to use the cores allocated by Slurm.

echo "Starting MPI job with $SLURM_NTASKS tasks..."

mpiexec -n $SLURM_NTASKS python -m mpi4py.futures scripts/fit_hpc.py \
  --data data/filtered_input1.csv \
  --ptm-intra data/ptm_intra.db \
  --ptm-inter data/ptm_inter.db \
  --kea-ks-table data/ks_psite_table.tsv \
  --unified-graph-pkl data/unified_kinase_graph.gpickle \
  --lambda-net 1e-3 \
  --outdir network_fit_mpi \
  --pop-size 400