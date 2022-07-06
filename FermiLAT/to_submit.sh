#!/bin/bash -l
#
# allocate 1 node (4 Cores) for 1 hours
#PBS -l nodes=1:ppn=4,walltime=23:59:00
# 
# job name
#PBS -N fermi_dataset_parallel
#
# first non-empty non-comment line ends PBS options
# load required modules (compiler, ...)
# module load intel64
# jobs always start in $HOME -
# change to work directory
export OMP_NUM_THREADS=4
conda activate gammapy-0.19

cd /home/vault/caph/mppi062h/repositories/GC/FermiLAT/scripts_parallel
python create_dataset.py
