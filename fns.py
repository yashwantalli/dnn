import numpy as np
import torch 
import torch.nn.functional as F


def compute_entropy(p):
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log2(p))

def kl_divergence(p, q):
    p = torch.clamp(p, 1e-10, 1.0)
    q = torch.clamp(q, 1e-10, 1.0)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=1).mean()

def kl_loss(targets, logits):
    log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, targets, reduction='batchmean')

def softmax(x):
    return torch.softmax(x, dim=1)

def entropy(p):
    p = torch.clamp(p, 1e-10, 1.0)
    return -torch.sum(p * torch.log2(p), dim=1)