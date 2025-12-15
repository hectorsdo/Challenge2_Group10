#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /fhome/vlia10/
#SBATCH -t 1-00:00          
#SBATCH -p tfg
#SBATCH --mem 16384         # 16GB RAM
#SBATCH -o /fhome/vlia10/extract.out
#SBATCH -e /fhome/vlia10/extract.err
#SBATCH --gres gpu:1

source /fhome/vlia10/env/bin/activate

python3 -u /fhome/vlia10/extract_features.py