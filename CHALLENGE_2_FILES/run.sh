#!/bin/bash
#SBATCH -n 4                # Solicita 4 núcleos de CPU (Máximo 10) [cite: 1437]
#SBATCH -N 1                # Asegura que todos los núcleos estén en la misma máquina [cite: 1438]
#SBATCH -D /fhome/vlia10/ # Directorio de trabajo (¡CAMBIA 'tu_usuario' POR EL TUYO!) [cite: 1506]
#SBATCH -t 1-00:00          # Tiempo límite: 0 días, 2 horas (Formato D-HH:MM) [cite: 1440]
#SBATCH -p tfg              # Partición a la que se envía el trabajo [cite: 1441]
#SBATCH --mem 12288         # Solicita 12GB de RAM (Máximo 60GB) [cite: 1442]
#SBATCH -o %x_%u_%j.out     # Archivo de salida estándar (Output) [cite: 1443]
#SBATCH -e %x_%u_%j.err     # Archivo de errores (Error) [cite: 1444]
#SBATCH --gres gpu:1        # Solicita 1 GPU (Máximo 8) [cite: 1443]

source /fhome/vlia10/env/bin/activate

# Ejecutar entrenamiento
python3 -u /fhome/vlia10/trainAE.py