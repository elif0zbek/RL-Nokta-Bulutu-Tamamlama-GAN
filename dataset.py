import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset

class ShapeNetDataset(Dataset):
    def __init__(self, root_dir, n_points=2048):
        self.n_points = n_points
        self.file_paths = glob.glob(os.path.join(root_dir, '**/*.pts'), recursive=True)
        if not self.file_paths:
            self.file_paths = glob.glob(os.path.join(root_dir, '*.*'))

    def __len__(self): 
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Dosyayı yükle
        points = np.loadtxt(self.file_paths[idx]).astype(np.float32)[:, 0:3]
        
        # Nokta sayısını sabitle (Resampling)
        if len(points) >= self.n_points:
            indices = np.random.choice(len(points), self.n_points, replace=False)
        else:
            indices = np.random.choice(len(points), self.n_points, replace=True)
        complete = points[indices]
        
        # Rastgele bir bakış açısı seç ve noktaların %60'ını tut (Occlusion)
        viewpoint = np.random.randn(3); viewpoint /= np.linalg.norm(viewpoint)
        projections = np.dot(complete, viewpoint)
        idx_sorted = np.argsort(projections)
        partial = complete[idx_sorted[:int(len(complete) * 0.6)]]
        
        # Padding (Eksik noktaları 0 ile doldur)
        if len(partial) < self.n_points:
            pad = np.zeros((self.n_points - len(partial), 3), dtype=np.float32)
            partial = np.concatenate([partial, pad], axis=0)
            
        return torch.from_numpy(partial), torch.from_numpy(complete)