# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# --- CONFIGURATION ---
# Carpeta donde se guardaron los vectores
FEATURES_DIR = '/fhome/vlia10/features_resnet152/'
# El Excel con las etiquetas
EXCEL_PATH = '/export/fhome/vlia/HelicoDataSet/HP_WSI-CoordAnnotatedAllPatches.xlsx'

print("--- SYSTEM 2: SUPERVISED CLASSIFICATION (SVM) ---")

# 1. PREPARAR DATOS (AGREGACION POR PACIENTE)
print(f"Reading labels from: {EXCEL_PATH}")
try:
    df = pd.read_excel(EXCEL_PATH)
except Exception as e:
    print(f"Error reading Excel: {e}")
    exit()

df.columns = df.columns.str.strip()
df = df[df['Presence'].isin([1, -1])] # Solo Sanos (-1) y Enfermos (1)

# Obtener lista unica de pacientes y su etiqueta real
patient_labels = df.groupby('Pat_ID')['Presence'].max().reset_index()
# Convertir -1 a 0 para que sea binario
patient_labels['Label'] = (patient_labels['Presence'] == 1).astype(int)

X = [] # Vectores de paciente (Features)
y = [] # Etiquetas (0 o 1)
ids = [] # IDs para trackear

print("Loading feature vectors (this is fast)...")
not_found = 0

for index, row in patient_labels.iterrows():
    pat_id = str(row['Pat_ID']).strip()
    label = row['Label']
    
    # Buscar la carpeta del paciente
    pat_dir = os.path.join(FEATURES_DIR, pat_id)
    
    # Busqueda flexible de carpeta
    if not os.path.exists(pat_dir):
        try:
            candidates = [d for d in os.listdir(FEATURES_DIR) if d.startswith(pat_id)]
            if len(candidates) > 0:
                pat_dir = os.path.join(FEATURES_DIR, candidates[0])
            else:
                not_found += 1
                continue
        except:
            continue

    # Cargar todos los .pt de ese paciente
    vectors = []
    try:
        if os.path.exists(pat_dir):
            for file in os.listdir(pat_dir):
                if file.endswith('.pt'):
                    v = torch.load(os.path.join(pat_dir, file))
                    vectors.append(v.numpy().flatten())
    except Exception as e:
        continue
        
    if len(vectors) == 0:
        continue
        
    # --- LA MAGIA: AGREGACION (MEAN POOLING) ---
    # Calculamos el vector PROMEDIO del paciente.
    patient_embedding = np.mean(vectors, axis=0)
    
    X.append(patient_embedding)
    y.append(label)
    ids.append(pat_id)

X = np.array(X)
y = np.array(y)

print(f"\nData loaded: {len(X)} patients.")
print(f"Patients not found (probably unannotated): {not_found}")

if len(X) < 10:
    print("Error: Too few data points to train. Check if extract_features finished correctly.")
    exit()

# 2. ENTRENAMIENTO (CROSS-VALIDATION)
print("\nStarting Cross-Validation (5-Fold)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_acc = []
scores_auc = []

fold = 1
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Modelo: SVM con probabilidad
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Predicciones
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    try:
        auc_val = roc_auc_score(y_test, y_proba)
    except:
        auc_val = 0.5
        
    scores_acc.append(acc)
    scores_auc.append(auc_val)
    
    print(f"Fold {fold}: Accuracy={acc:.4f}, AUC={auc_val:.4f}")
    fold += 1

# 3. RESULTADOS FINALES
print("\n" + "="*40)
print(f"FINAL RESULT SYSTEM 2 (SVM):")
print(f"AVERAGE ACCURACY: {np.mean(scores_acc):.4f}")
print(f"AVERAGE AUC:      {np.mean(scores_auc):.4f}")
print("="*40)