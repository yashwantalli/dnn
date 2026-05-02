
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10H_Dataset(Dataset):
    def __init__(self, indices, testset, soft_labels):
        self.indices = indices
        self.testset = testset
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        image, _ = self.testset[real_idx]
        label_dist = self.soft_labels[real_idx]
        label_dist = torch.tensor(label_dist, dtype=torch.float32)
        label_dist = label_dist / label_dist.sum()  # safety
        
        return image, label_dist

