# src/eeg_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import argparse
import mne

from model import UNet1D
# MODIFICATION: Import get_adjacency_list
from eeg_data_utils import load_eeg_from_edf, create_eeg_segments, EEGDataset, TARGET_FS, get_adjacency_list

def find_common_channels(directory):
    # ... (this function remains the same, no changes needed)
    common_channels = None
    print(f"Finding common channels in: {directory}")
    files_to_check = [f for f in os.listdir(directory) if f.endswith('.edf')]
    if not files_to_check:
        raise FileNotFoundError(f"No .edf files found in {directory}. Please check the path.")
    for f in tqdm(files_to_check, desc="Scanning file headers"):
        file_path = os.path.join(directory, f)
        try:
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            current_channels = set(ch.upper() for ch in raw.ch_names)
            if common_channels is None:
                common_channels = current_channels
            else:
                common_channels.intersection_update(current_channels)
        except Exception as e:
            print(f"Could not read header from {f}: {e}")
    return list(common_channels)

# --- NEW LOSS FUNCTIONS ---
class TemporalGradientLoss(nn.Module):
    def __init__(self):
        super(TemporalGradientLoss, self).__init__()
        self.loss = nn.L1Loss()
    def forward(self, pred, target):
        pred_grad = torch.diff(pred, dim=-1)
        target_grad = torch.diff(target, dim=-1)
        return self.loss(pred_grad, target_grad)

class LaplacianLoss(nn.Module):
    def __init__(self, adj_list):
        super(LaplacianLoss, self).__init__()
        self.adj_list = adj_list
        self.loss = nn.L1Loss()
    
    def _calculate_laplacian(self, x):
        # x shape: [B, C, T]
        L = torch.zeros_like(x)
        for i, neighbors in enumerate(self.adj_list):
            if len(neighbors) > 0:
                mean_neighbors = torch.mean(x[:, neighbors, :], dim=1)
                L[:, i, :] = x[:, i, :] - mean_neighbors
        return L

    def forward(self, pred, target):
        lap_pred = self._calculate_laplacian(pred)
        lap_target = self._calculate_laplacian(target)
        return self.loss(lap_pred, lap_target)
# -------------------------

class Config:
    # ... (this class remains the same)
    DEFAULT_DATA_DIR = "data/chb-mit-scalp-eeg-database-1.0.0/"
    DEFAULT_MODEL_SAVE_PATH = "models/eeg_denoiser_base.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    NUM_WORKERS = 2
    WINDOW_SECONDS = 2
    WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_FS)
    SAMPLES_PER_EPOCH = 5000

def train_one_epoch(loader, model, optimizer, loss_fns, scaler, config, args):
    loop = tqdm(loader, leave=True)
    total_loss = 0.0
    for noisy, clean in loop:
        noisy = noisy.to(config.DEVICE)
        clean = clean.to(config.DEVICE)
        
        with torch.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=config.DEVICE=='cuda'):
            denoised = model(noisy)
            
            # --- MODIFICATION: Calculate loss based on experiment ---
            loss_amp = loss_fns['L1'](denoised, clean)
            
            if args.experiment == 'spatial':
                loss_temp = loss_fns['Temporal'](denoised, clean)
                loss_spat = loss_fns['Spatial'](denoised, clean)
                loss = loss_amp + (args.alpha * loss_temp) + (args.beta * loss_spat)
            else: # Baseline
                loss = loss_amp
            # --------------------------------------------------------

        if config.DEVICE == 'cuda':
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def main(args):
    config = Config()
    print(f"--- Starting EEG Denoiser Training ---")
    print(f"Device: {config.DEVICE}, Experiment: {args.experiment}")
    
    subject_dir = os.path.join(args.data_dir, 'chb01')
    common_channels = find_common_channels(subject_dir)
    config.NUM_CHANNELS = len(common_channels)
    print(f"Found {config.NUM_CHANNELS} common channels: {common_channels}")
    
    # --- MODIFICATION: Get adjacency list for loss function ---
    adjacency_list = get_adjacency_list(common_channels)
    # ---------------------------------------------------------
    
    all_segments = []
    # ... (data loading remains the same)
    for f in tqdm(os.listdir(subject_dir), desc="Loading and segmenting files"):
        if f.endswith('.edf'):
            file_path = os.path.join(subject_dir, f)
            data = load_eeg_from_edf(file_path, desired_channels=common_channels)
            if data is not None:
                segments = create_eeg_segments(data, config.WINDOW_SAMPLES)
                all_segments.extend(segments)
    
    train_dataset = EEGDataset(clean_segments=all_segments, samples_per_epoch=config.SAMPLES_PER_EPOCH)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                              pin_memory=(config.DEVICE=='cuda'), shuffle=True)
    
    model = UNet1D(in_channels=config.NUM_CHANNELS, out_channels=config.NUM_CHANNELS).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE=='cuda'))
    
    # --- MODIFICATION: Create a dictionary of loss functions ---
    loss_functions = {
        'L1': nn.L1Loss()
    }
    if args.experiment == 'spatial':
        loss_functions['Temporal'] = TemporalGradientLoss()
        loss_functions['Spatial'] = LaplacianLoss(adjacency_list)
    # -----------------------------------------------------------

    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    for epoch in range(config.NUM_EPOCHS):
        avg_loss = train_one_epoch(train_loader, model, optimizer, loss_functions, scaler, config, args)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Average Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), args.model_save_path)

    print(f"\n✅ Training complete. Model saved to {args.model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 1D U-Net for Multi-Channel EEG Denoising.')
    parser.add_argument('--experiment', type=str, default='baseline',
                        choices=['baseline', 'spatial', 'frequency', 'self_supervised'])
    parser.add_argument('--data_dir', type=str, default=Config.DEFAULT_DATA_DIR)
    parser.add_argument('--model_save_path', type=str, default=Config.DEFAULT_MODEL_SAVE_PATH)
    # --- MODIFICATION: Add alpha and beta for weighting the spatial losses ---
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for temporal gradient loss.')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for spatial laplacian loss.')
    # --------------------------------------------------------------------
    cli_args = parser.parse_args()
    main(cli_args)