# Challenge2_Group10
# H. pylori Infection Diagnosis from Gastric Biopsies (WSI)

This repository contains the source code for the "H. pylori Detection Challenge". The project implements two distinct approaches to diagnose *Helicobacter pylori* infection using Whole Slide Images (WSI) without pixel-level annotations.

##Project Overview

We propose a dual approach to handle the lack of local annotations and data inconsistencies:

* **System 1 (Unsupervised):** Anomaly Detection using a Convolutional Autoencoder (CAE). It reconstructs healthy tissue and identifies infection based on the reconstruction error (FRED score) in the Red channel. Includes a **Brute-Force Indexing** strategy to recover data from the server.
* **System 2 (Weakly Supervised):** A Multiple Instance Learning (MIL) pipeline. It extracts features using a pre-trained **ResNet-152**, aggregates them via Global Average Pooling, and classifies patients using a **Support Vector Machine (SVM)**.

---

##Requirements

* Python 3.8+
* PyTorch & Torchvision
* Scikit-learn
* Pandas
* Numpy
* Matplotlib
* Pillow (PIL)

---

## Usage Instructions

The code is designed to run on a Slurm-based HPC cluster.

### System 1: Unsupervised Anomaly Detection

**1. Training (Optional)**
The model is trained on healthy patches to learn the normal tissue manifold.

sbatch run.sh
# Runs: trainAE.py


Usage Instructions

System 1: Unsupervised Anomaly Detection
1. Training (Optional) The model is trained on healthy patches to learn the normal tissue manifold.
sbatch run.sh
# Runs: trainAE.py

2. Evaluation & Data Recovery We use a brute-force indexing script to locate images on the server and calculate anomaly scores.
sbatch run_eval_full.sh
# Runs: eval_system1_full.py
# Output: resultados_completo_sistema1.csv

3. Patient Diagnosis & Metrics Once the CSV is generated, run this script to optimize the threshold and obtain final Patient-Level metrics.
python3 patient_diagnosis_opt.py

System 2: Weakly Supervised Classification
1. Feature Extraction Extracts latent vectors (2048-dim) from WSI patches using ResNet-152.
sbatch run_extract.sh
# Runs: extract_features.py

2. Training & Cross-Validation Trains the SVM classifier using 5-Fold Stratified Cross-Validation and generates performance metrics. (See Visualization section below)
Visualization (Results)
To generate Confusion Matrices and Boxplots for the report (saved as .png files), execute:
sbatch run_plots.sh
# Runs: plot_system1.py and plot_system2.py

Output files:

confusion_matrix_system1.png

boxplot_scores_system1.png

confusion_matrix_system2.png

boxplot_auc_system2.png
