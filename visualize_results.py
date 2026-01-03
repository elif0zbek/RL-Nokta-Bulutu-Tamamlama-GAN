import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import glob

# Modüler yapıdan importlar
from models import PointNetGenerator
from dataset import ShapeNetDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data"
MODEL_DIR = "./models/V3_Interleaved"
OUTPUT_ROOT = "./Sonuc_Raporu"
DIR_2D = os.path.join(OUTPUT_ROOT, "2D_Gorseller")
DIR_3D = os.path.join(OUTPUT_ROOT, "3D_Gorseller")

os.makedirs(DIR_2D, exist_ok=True)
os.makedirs(DIR_3D, exist_ok=True)

def save_plot_3d(partial, predicted, complete, sample_id, epoch_num):
    fig = plt.figure(figsize=(18, 6))
    p_clean = partial[~np.all(partial == 0, axis=1)]
    data_list = [p_clean, predicted, complete]
    titles = ["Eksik Girdi", f"Model Tahmini (Ep {epoch_num})", "Orijinal"]
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        d = data_list[i]
        ax.scatter(d[:, 0], d[:, 2], d[:, 1], s=2, c=colors[i], alpha=0.6)
        ax.set_title(titles[i])
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)
    
    plt.savefig(os.path.join(DIR_3D, f"Ornek_{sample_id}_3D.png"), dpi=150, bbox_inches='tight')
    plt.close()

def save_plot_2d(partial, predicted, complete, sample_id, epoch_num):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    p_clean = partial[~np.all(partial == 0, axis=1)]
    data_list = [p_clean, predicted, complete]
    titles = ["Eksik (2D)", "Tahmin (2D)", "Orijinal (2D)"]
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        axes[i].scatter(data_list[i][:, 0], data_list[i][:, 2], s=4, c=colors[i], alpha=0.6)
        axes[i].set_title(titles[i])
        axes[i].set_aspect('equal')
        axes[i].axis('off')
        
    plt.savefig(os.path.join(DIR_2D, f"Ornek_{sample_id}_2D.png"), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # --- MODEL SIRALAMA VE YÜKLEME ---
    model_files = glob.glob(os.path.join(MODEL_DIR, "ajan_epoch_*.pth"))
    if not model_files:
        print("HATA: Model dosyası bulunamadı!")
        exit()
    
    # Sayısal değere göre en sonuncuyu bul (ajan_epoch_10.pth gibi)
    model_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    latest_model = model_files[-1]
    epoch_num = os.path.basename(latest_model).split('_')[-1].replace(".pth", "")
    
    print(f"Kullanılan Model: {latest_model}")

    actor = PointNetGenerator().to(DEVICE)
    actor.load_state_dict(torch.load(latest_model, map_location=DEVICE))
    actor.eval()

    dataset = ShapeNetDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    iterator = iter(dataloader)
    print("Görseller üretiliyor...")
    for i in range(1, 4):
        try:
            partial, complete = next(iterator)
            with torch.no_grad():
                predicted = actor(partial.to(DEVICE))
            
            p_np = partial[0].cpu().numpy()
            pr_np = predicted[0].cpu().numpy()
            c_np = complete[0].cpu().numpy()
            
            save_plot_3d(p_np, pr_np, c_np, i, epoch_num)
            save_plot_2d(p_np, pr_np, c_np, i, epoch_num)
            print(f" Örnek {i} tamamlandı.")
        except StopIteration:
            break

    print(f"\nİşlem bitti. Sonuçlar '{OUTPUT_ROOT}' klasöründe.")