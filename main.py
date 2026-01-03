import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Kendi dosyalarımızdan modelleri ve dataseti çağırıyoruz
from models import PointNetGenerator, PointNetDiscriminator
from dataset import ShapeNetDataset

# --- YAPILANDIRMA ---s
DATA_PATH = "./data" 
SAVE_DIR = "./models/V3_Interleaved"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

def chamfer_distance(p1, p2):
    """Basit Chamfer Distance hesaplaması"""
    x, y = p1.unsqueeze(2), p2.unsqueeze(1)
    dist = torch.sum((x - y) ** 2, dim=-1)
    return torch.mean(torch.min(dist, dim=2)[0]) + torch.mean(torch.min(dist, dim=1)[0])

def train(epochs, is_minitest=False):
    print(f"--- {'MINITEST' if is_minitest else 'FULL TRAINING'} BAŞLIYOR ---")
    print(f"Cihaz: {DEVICE}")
    
    dataset = ShapeNetDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=8 if is_minitest else 16, shuffle=True)
    
    actor = PointNetGenerator().to(DEVICE)
    critic = PointNetDiscriminator().to(DEVICE)
    
    opt_G = optim.Adam(actor.parameters(), lr=0.0002)
    opt_D = optim.Adam(critic.parameters(), lr=0.0002)
    criterion_GAN = nn.BCELoss()
    
    ALPHA = 200 # Geometri kaybı ağırlığı
    CYCLE_LEN = 10
    GAN_START = 5 

    for epoch in range(epochs):
        cycle_pos = epoch % CYCLE_LEN
        is_gan_mode = cycle_pos >= GAN_START
        mode_desc = "GAN" if is_gan_mode else "GEOMETRY"
        pbar = tqdm(loader, desc=f"Ep {epoch+1}/{epochs} [{mode_desc}]")
        
        for partial, complete in pbar:
            partial, complete = partial.to(DEVICE), complete.to(DEVICE)
            B = partial.size(0)
            
            # --- DISCRIMINATOR GÜNCELLEME ---
            d_loss = torch.tensor(0.0).to(DEVICE)
            if is_gan_mode:
                opt_D.zero_grad()
                real_labels = torch.ones(B, 1).to(DEVICE)
                fake_labels = torch.zeros(B, 1).to(DEVICE)
                
                d_real = criterion_GAN(critic(complete), real_labels)
                d_fake = criterion_GAN(critic(actor(partial).detach()), fake_labels)
                d_loss = (d_real + d_fake) / 2
                d_loss.backward()
                opt_D.step()

            # --- GENERATOR GÜNCELLEME ---
            opt_G.zero_grad()
            fake_cloud = actor(partial)
            g_cd = chamfer_distance(fake_cloud, complete)
            
            if is_gan_mode:
                g_adv = criterion_GAN(critic(fake_cloud), torch.ones(B, 1).to(DEVICE))
                g_loss = g_adv + (ALPHA * g_cd)
            else:
                g_loss = g_cd * ALPHA
                
            g_loss.backward()
            opt_G.step()
            pbar.set_postfix(G_Loss=f"{g_loss.item():.4f}", D_Loss=f"{d_loss.item():.4f}")

        # Her 5 epochta bir veya minitest sonunda kaydet
        if (epoch + 1) % 5 == 0 or is_minitest:
            path = f"{SAVE_DIR}/ajan_epoch_{epoch+1}.pth"
            torch.save(actor.state_dict(), path)
            print(f" Model kaydedildi: {path}")

if __name__ == "__main__":
    # Test için 10 epoch çalıştırıyoruz
    train(epochs=10, is_minitest=True)