# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- CONFIGURACION ---
# Ruta de las imagenes originales
DATA_DIR = '/export/fhome/vlia/HelicoDataSet/CrossValidation/Cropped/'
# Donde guardaremos los vectores (features)
OUTPUT_DIR = '/fhome/vlia10/features_resnet152/'
BATCH_SIZE = 64 

# --- 1. MODELO BACKBONE (ResNet-152) ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Usamos ResNet152 pre-entrenada
        self.resnet = models.resnet152(pretrained=True)
        # Quitamos la ultima capa
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)

# --- 2. DATASET ---
class ImageListDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        # Buscar todas las imagenes recursivamente
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.image_paths.append(os.path.join(root, file))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except:
            img = torch.zeros(3, 224, 224) # Dummy si falla
        return img, path

# --- 3. EXTRACCION ---
def extract():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Transformaciones necesarias para ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageListDataset(DATA_DIR, transform=transform)
    # num_workers=4 es clave para ir rapido
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Iniciando extraccion de {len(dataset)} imagenes...")
    
    model = FeatureExtractor().to(device)
    model.eval()

    with torch.no_grad():
        for i, (imgs, paths) in enumerate(dataloader):
            imgs = imgs.to(device)
            features = model(imgs) # Saca vector de 2048 numeros
            
            # Guardar cada vector individualmente
            for j in range(len(paths)):
                feature_vector = features[j].cpu()
                original_path = paths[j]
                
                parts = original_path.split(os.sep)
                patient_folder = parts[-2] 
                filename = parts[-1].replace('.png', '.pt').replace('.jpg', '.pt')
                
                save_dir = os.path.join(OUTPUT_DIR, patient_folder)
                os.makedirs(save_dir, exist_ok=True)
                
                torch.save(feature_vector, os.path.join(save_dir, filename))

            if (i+1) % 50 == 0:
                print(f"Lote {i+1} procesado...")

    print("Extraccion completada!")

if __name__ == '__main__':
    extract()