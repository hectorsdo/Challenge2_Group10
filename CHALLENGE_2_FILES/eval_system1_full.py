# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_curve, auc

# --- CONFIGURATION ---
MODEL_PATH = '/fhome/vlia10/autoencoder_quiron.pth' 
BASE_DIR = '/export/fhome/vlia/HelicoDataSet/CrossValidation/Cropped/'
EXCEL_PATH = '/export/fhome/vlia/HelicoDataSet/HP_WSI-CoordAnnotatedAllPatches.xlsx'

# --- MODEL ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def calculate_fred(original, reconstructed):
    ori_np = original.permute(1, 2, 0).cpu().numpy()
    rec_np = reconstructed.permute(1, 2, 0).cpu().numpy()
    ori_hsv = mcolors.rgb_to_hsv(ori_np)
    rec_hsv = mcolors.rgb_to_hsv(rec_np)
    lower, upper = 0.055, 0.945
    mask_ori = (ori_hsv[:,:,0] < lower) | (ori_hsv[:,:,0] > upper)
    mask_rec = (rec_hsv[:,:,0] < lower) | (rec_hsv[:,:,0] > upper)
    return mask_ori.sum() / (mask_rec.sum() + 1e-8)

# --- FILE INDEXER (BRUTE FORCE) ---
def index_all_files(root_dir):
    print("Indexing all files on server (this may take a while)...")
    file_map = {} 
    
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                try:
                    # 1. Get Window ID (e.g., '29.png' -> '29')
                    w_id_str = os.path.splitext(file)[0]
                    w_id = str(int(w_id_str)) # Remove zeros: '0029' -> '29'
                    
                    # 2. Get Patient ID from folder (e.g., 'B22-240_1' -> 'B22-240')
                    folder_name = os.path.basename(root)
                    if "_" in folder_name:
                        pat_id = folder_name.split("_")[0]
                    else:
                        pat_id = folder_name
                    
                    # 3. Store in map
                    key = (pat_id, w_id)
                    
                    if key not in file_map:
                        file_map[key] = os.path.join(root, file)
                        count += 1
                except:
                    continue
    
    print(f"Indexing complete: {count} images located.")
    return file_map

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = Autoencoder().to(device)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            print("Model loaded.")
        except:
            print("Model corrupt.")
            return
    else:
        print("Model not found.")
        return

    # RUN INDEXER
    file_map = index_all_files(BASE_DIR)

    # Read Excel
    print(f"Reading Excel: {EXCEL_PATH}")
    try:
        df = pd.read_excel(EXCEL_PATH)
    except:
        print("Error reading Excel.")
        return

    df.columns = df.columns.str.strip()
    df = df[df['Presence'].isin([1, -1])]
    
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    y_true = []
    y_scores = []
    y_pat_ids = [] 
    found_count = 0

    print(f"Evaluating {len(df)} patches against index...")
    
    with torch.no_grad():
        for index, row in df.iterrows():
            if index % 200 == 0:
                print(f"Processing row {index}...")

            # Clean IDs from Excel
            pat_id = str(row['Pat_ID']).strip()
            
            try:
                window_id = str(int(row['Window_ID']))
            except:
                window_id = str(row['Window_ID']).strip()
            
            # LOOKUP IN MAP (Instant)
            key = (pat_id, window_id)
            
            if key in file_map:
                img_path = file_map[key]
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_t = transform(img).unsqueeze(0).to(device)
                    recon = model(img_t)
                    score = calculate_fred(img_t[0], recon[0])
                    
                    label = 1 if row['Presence'] == 1 else 0
                    y_true.append(label)
                    y_scores.append(score)
                    y_pat_ids.append(pat_id)
                    found_count += 1
                except:
                    pass
            
    print(f"Evaluation finished.")
    print(f"Found: {found_count} / {len(df)}")

    # Save results
    results_df = pd.DataFrame({
        'Pat_ID': y_pat_ids,
        'Label_Real': y_true, 
        'Fred_Score': y_scores
    })
    results_df.to_csv('/fhome/vlia10/resultados_completo_sistema1.csv', index=False)
    
    if len(y_scores) > 0:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"--- AUC PATCH LEVEL (FULL): {roc_auc:.4f} ---")

if __name__ == '__main__':
    evaluate()