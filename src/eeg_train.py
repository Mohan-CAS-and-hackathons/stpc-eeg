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

# Import our EEG-specific modules
from model import UNet1D
from eeg_data_utils import load_eeg_from_edf, create_eeg_segments, EEGDataset, TARGET_FS

def find_common_channels(directory):
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

class Config:
    # These are now defaults, can be overridden by command-line args
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

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, config):
    loop = tqdm(loader, leave=True)
    total_loss = 0.0
    for noisy, clean in loop:
        noisy, clean = noisy.to(config.DEVICE)
        clean = clean.to(config.DEVICE)
        with torch.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=config.DEVICE=='cuda'):
            denoised = model(noisy)
            loss = loss_fn(denoised, clean)
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
    print(f"Data Directory: {args.data_dir}")
    print(f"Model will be saved to: {args.model_save_path}")

    subject_dir = os.path.join(args.data_dir, 'chb01')
    common_channels = find_common_channels(subject_dir)
    config.NUM_CHANNELS = len(common_channels)
    print(f"Found {config.NUM_CHANNELS} common channels. Using these for training.")
    
    all_segments = []
    for f in tqdm(os.listdir(subject_dir), desc="Loading and segmenting files"):
        if f.endswith('.edf'):
            file_path = os.path.join(subject_dir, f)
            data = load_eeg_from_edf(file_path, desired_channels=common_channels)
            if data is not None:
                segments = create_eeg_segments(data, config.WINDOW_SAMPLES)
                all_segments.extend(segments)
    
    train_dataset = EEGDataset(clean_segments=all_segments, samples_per_epoch=config.SAMPLES_PER_EPOCH)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE=='cuda'), shuffle=True
    )
    
    model = UNet1D(in_channels=config.NUM_CHANNELS, out_channels=config.NUM_CHANNELS).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE=='cuda'))

    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    for epoch in range(config.NUM_EPOCHS):
        avg_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Average Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), args.model_save_path)

    print(f"\n✅ Training complete. Model saved to {args.model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 1D U-Net for Multi-Channel EEG Denoising.')
    parser.add_argument('--experiment', type=str, default='baseline',
                        choices=['baseline', 'spatial', 'frequency', 'self_supervised'])
    # --- KEY MODIFICATION: Added command-line arguments for paths ---
    parser.add_argument('--data_dir', type=str, default=Config.DEFAULT_DATA_DIR,
                        help='Path to the root EEG data directory.')
    parser.add_argument('--model_save_path', type=str, default=Config.DEFAULT_MODEL_SAVE_PATH,
                        help='Path to save the trained model.')
    cli_args = parser.parse_args()
    main(cli_args)