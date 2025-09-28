# src/stpc/inference.py
import torch
import numpy as np
from .model import UNet1D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_denoiser_model(model_path, device=DEVICE):
    model = UNet1D(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    return model

def denoise_ecg_signal(noisy_signal_np, model, device=DEVICE):
    noisy_tensor = torch.from_numpy(noisy_signal_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)
    denoised_signal_np = denoised_tensor.squeeze(0).squeeze(0).cpu().numpy()
    return denoised_signal_np