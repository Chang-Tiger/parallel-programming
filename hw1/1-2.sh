#!/bin/bash
#SBATCH -n 6
#SBATCH -N 2
echo "N2 n6"
srun time ./hw1 536869888 /home/pp23/share/hw1/testcases/35.in ./out