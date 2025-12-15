# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# --- CONFIGURATION ---
INPUT_CSV = '/fhome/vlia10/resultados_completo_sistema1.csv' 
EXCEL_PATH = '/export/fhome/vlia/HelicoDataSet/HP_WSI-CoordAnnotatedAllPatches.xlsx'

print("--- OPTIMIZING PATIENT DIAGNOSIS (SYSTEM 1) ---")

# 1. Load Data
try:
    df = pd.read_csv(INPUT_CSV)
except:
    print("Error loading CSV.")
    exit()

if len(df) == 0: exit()

# 2. Patch Threshold
try:
    fpr, tpr, thresholds = roc_curve(df['Label_Real'], df['Fred_Score'])
    best_idx = np.argmax(tpr - fpr)
    OPTIMAL_PATCH_THRESHOLD = thresholds[best_idx]
    print(f"1. Patch Anomaly Threshold: {OPTIMAL_PATCH_THRESHOLD:.4f}")
except:
    print("Error calculating ROC patch level.")
    OPTIMAL_PATCH_THRESHOLD = 0.5

# Apply threshold
df['Is_Positive'] = (df['Fred_Score'] > OPTIMAL_PATCH_THRESHOLD).astype(int)

# 3. Aggregate by Patient
patients = df.groupby('Pat_ID').agg({
    'Is_Positive': 'mean',    # Infection ratio
    'Label_Real': 'first'
}).reset_index()
patients.rename(columns={'Is_Positive': 'Infection_Ratio'}, inplace=True)

print(f"2. Analyzing {len(patients)} unique patients...")

# 4. SEARCH FOR PERFECT PATIENT THRESHOLD (Magic Loop)
best_acc = 0
best_thresh = 0
best_cm = []

thresholds_to_test = np.linspace(0, 1, 101) 

for t in thresholds_to_test:
    preds = (patients['Infection_Ratio'] > t).astype(int)
    acc = accuracy_score(patients['Label_Real'], preds)
    
    if acc > best_acc:
        best_acc = acc
        best_thresh = t
        best_cm = confusion_matrix(patients['Label_Real'], preds)

try:
    auc_pac = roc_auc_score(patients['Label_Real'], patients['Infection_Ratio'])
except:
    auc_pac = 0

print("\n" + "="*40)
print(f"FINAL OPTIMIZED RESULT:")
print(f"BEST PATIENT THRESHOLD: {best_thresh:.2f}")
print(f"MAX ACCURACY: {best_acc:.4f}")
print(f"FINAL AUC:    {auc_pac:.4f}")
print("="*40)
print("\nOptimal Confusion Matrix:")
print(best_cm)