#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/vlia10/
#SBATCH -t 0-10:00
#SBATCH -p tfg
#SBATCH --mem 12000
#SBATCH -o /fhome/vlia10/eval_full.out
#SBATCH -e /fhome/vlia10/eval_full.err
#SBATCH --gres gpu:1

source /fhome/vlia10/env/bin/activate

python3 -u /fhome/vlia10/eval_system1_full.py