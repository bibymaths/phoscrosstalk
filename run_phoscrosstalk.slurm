#!/bin/bash
#SBATCH --job-name=phoscrosstalk_pymoo
#SBATCH --partition=SiCore
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# 1) Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phoscrosstalk

# 2) Go to the *correct* project directory
cd /home/abhinav/Documents/phoscrosstalk

# 3) Ensure logs dir exists (in case you submit from elsewhere)
#mkdir -p logs

# 4) Avoid BLAS oversubscribing cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 5) Run the model with unbuffered output (-u) so tail -f sees logs immediately
srun --mpi=pmix python -u scripts/fit_hpc.py \
  --data data/input1.csv \
  --ptm-intra data/ptm_intra.db \
  --ptm-inter data/ptm_inter.db \
  --kea-ks-table data/ks_psite_table.tsv \
  --unified-graph-pkl data/unified_kinase_graph.gpickle \
  --lambda-net 1e-3 \
  --outdir network_fit_all