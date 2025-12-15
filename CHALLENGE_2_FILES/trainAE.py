import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- CONFIGURACIÓN ---
# Ruta EXACTA del servidor según el PDF 
DATASET_PATH = '/export/fhome/vlia/HelicoDataSet/CrossValidation/Cropped/'
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100  # Puedes subirlo a 50 o 100 si tienes tiempo
LATENT_DIM = 64  # Dimensión del cuello de botella (bottleneck)

# --- 1. DATASET ---
class QuironDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Usamos os.walk para entrar en TODAS las subcarpetas (B22-01, B22-02, etc.)
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # Aceptamos varios formatos por si acaso (png, jpg, tif)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    full_path = os.path.join(root, file)
                    self.image_paths.append(full_path)
        
        # Debug: Para que veas en el archivo .out si está funcionando
        print(f"DEBUG: Buscando en: {root_dir}")
        print(f"DEBUG: ¡Encontradas {len(self.image_paths)} imágenes!")
        
        if len(self.image_paths) == 0:
            print("ERROR CRÍTICO: No se han encontrado imágenes. Revisa la ruta.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Abrir imagen y convertir a RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Si falla una imagen, devolvemos una negra (parche de seguridad)
            image = Image.new('RGB', (256, 256))
            
        if self.transform:
            image = self.transform(image)
        
        # En Autoencoders, el label es la misma imagen (reconstrucción)
        return image, image

# Transformaciones: Reducir tamaño para ir rápido y convertir a Tensor
data_transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
])

# --- 2. MODELO AUTOENCODER (Basado en Challenge_AEs.pptx) ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder [cite: 786, 789]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> [32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> [64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> [128, 16, 16]
            nn.ReLU()
        )
        
        # Decoder [cite: 793, 795]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Salida entre 0 y 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- 3. ENTRENAMIENTO ---
def main():
    # Configurar dispositivo (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar datos
    dataset = QuironDataset(root_dir=DATASET_PATH, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Iniciar modelo
    model = Autoencoder().to(device)
    
    
    
    modelo_previo = '/fhome/vlia10/autoencoder_quiron.pth'
    if os.path.exists(modelo_previo):
        print(f"Cargando pesos del modelo anterior: {modelo_previo}")
        model.load_state_dict(torch.load(modelo_previo))
    else:
        print("No se encontró modelo previo, empezando desde cero.")
    
    
    
    
    criterion = nn.MSELoss() # Loss para reconstrucción [cite: 881]
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Iniciando entrenamiento...")
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for data in dataloader:
            img, _ = data # Ignoramos etiquetas
            img = img.to(device)
            
            # Forward
            output = model(img)
            loss = criterion(output, img)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), '/fhome/vlia10/autoencoder_quiron.pth')
    print("Modelo guardado como '/fhome/vlia10/autoencoder_quiron.pth'")

if __name__ == '__main__':
    main()