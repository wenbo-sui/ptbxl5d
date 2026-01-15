import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# ================== SE block ==================
class SEblock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# ==================  U-Net SE ==================
class UNet_SE(nn.Module):
    def __init__(self, embedding_size=128):
        super(UNet_SE, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7), # -> 500
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3), # -> 250
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            SEblock(128)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1), # -> 125
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        
        # bottleneck
        self.bottleneck_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_latent = nn.Linear(256, embedding_size)
        self.fc_decoder = nn.Linear(embedding_size, 256 * 125)
        self.dec3 = nn.ConvTranspose1d(256 + 256, 128, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose1d(128 + 128, 64, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose1d(64 + 64, 12, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1) 
        e3 = self.enc3(e2) 
        
        # bottleneck
        latent = self.fc_latent(self.bottleneck_pool(e3).squeeze(-1))
        
        # decoder
        d3 = self.fc_decoder(latent).view(-1, 256, 125)
        d3 = torch.cat([d3, e3], dim=1)  
        d2 = F.relu(self.dec3(d3))
        d2 = torch.cat([d2, e2], dim=1) 
        d1 = F.relu(self.dec2(d2))
        d1 = torch.cat([d1, e1], dim=1) 
        
        out = self.dec1(d1) 
        return out, latent

# ================== Loss Function ==================
class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # A. Time Domain Loss (numerical alignment)
        mse_loss = self.mse(pred, target)
        
        # Frequency Domain Loss (to address smoothing issues)
        pred_flat = pred.view(-1, 1000)
        target_flat = target.view(-1, 1000)
        p_spec = torch.stft(pred_flat, n_fft=256, return_complex=True)
        t_spec = torch.stft(target_flat, n_fft=256, return_complex=True)
        spec_loss = torch.mean(torch.abs(p_spec.abs() - t_spec.abs()))
        
        # Morphological Correlation Loss 
        pred_c = pred - pred.mean(dim=-1, keepdim=True)
        target_c = target - target.mean(dim=-1, keepdim=True)
        corr = (pred_c * target_c).sum(dim=-1) / (
            torch.sqrt((pred_c**2).sum(dim=-1) * (target_c**2).sum(dim=-1) + 1e-8)
        )
        corr_loss = 1 - corr.mean()
        
        # Gradient Loss
        grad_loss = self.mse(torch.diff(pred, dim=-1), torch.diff(target, dim=-1))
        
        return mse_loss + 0.1 * spec_loss + 0.5 * corr_loss + 0.2 * grad_loss

