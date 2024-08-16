#!/bin/bash
#SBATCH -n 12
#SBATCH -N 3
echo "N3 n12"
srun time ./hw1 536869888 /home/pp23/share/hw1/testcases/35.in ./out