#!/bin/sh
#SBATCH -N 1
#SBATCH -J bv-ff
#SBATCH -p phi
#SBATCH --ntasks-per-node=64 # for xeon phi
export Fl_PROVIDER=tcp # for xeon phi

# The per-frame loss loop is parallelized over processes via joblib (controls
# fitting.n_jobs). Pin each worker's BLAS to a single thread so the joblib
# processes don't oversubscribe the cores by also spawning BLAS thread pools.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

eval "$(/home/sbpark/archive/anaconda3/condabin/conda shell.bash hook)"
conda activate env

python3 /archive/sbpark/git/M3L/bv-ff/src/main.py 
