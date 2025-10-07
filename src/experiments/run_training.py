# src/experiments/run_training.py
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# --- SETUP: Add project root to system path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import from our STPC library ---
from src.stpc.model import UNet1D, ECGClassifier
from src.stpc.losses import GradientLoss, FFTLoss, LaplacianLoss, BandMaskedFFTLoss
from src.stpc.data import PhysioNetDataset, ECGBeatDataset, EEGDataset, MaskedEEGDataset
from src.stpc.utils.ecg_utils import get_all_record_names, get_noise_signals, load_all_beats_from_dataset, load_and_resample_signal as load_ecg_signal
from src.stpc.utils.eeg_utils import load_eeg_from_edf, create_eeg_segments, get_adjacency_list, find_common_monopolar_channels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_ecg_denoiser(cfg: DictConfig):
    print("--- Training ECG Denoiser ---")
    print(f"Loaded configuration for experiment: {cfg.experiment_name}")
    print(OmegaConf.to_yaml(cfg))
    
    os.makedirs(os.path.dirname(cfg.paths.save_path), exist_ok=True)
    
    model = UNet1D(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training_params.learning_rate)
    
    all_record_names = get_all_record_names(cfg.paths.data_dir)
    noise_signals = get_noise_signals(cfg.paths.noise_dir, target_fs=250)

    loss_recon, loss_grad, loss_fft = nn.L1Loss(), GradientLoss(), FFTLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))

    for epoch in range(cfg.training_params.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.training_params.epochs}: Loading data...")
        records_for_epoch = np.random.choice(all_record_names, 20, replace=False)
        clean_signals = [sig for name in tqdm(records_for_epoch, desc="Loading signals") 
                         if (sig := load_ecg_signal(os.path.join(cfg.paths.data_dir, name), 250)) is not None 
                         and len(sig) > cfg.training_params.segment_samples]

        if not clean_signals: continue

        train_dataset = PhysioNetDataset(clean_signals, noise_signals, 
                                       OmegaConf.to_container(cfg.training_params, resolve=True), 5000)
        train_loader = DataLoader(train_dataset, batch_size=cfg.training_params.batch_size, 
                                  shuffle=True, num_workers=2, pin_memory=True)
        
        loop = tqdm(train_loader, desc="Training")
        for noisy, clean in loop:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE=='cuda')):
                denoised = model(noisy)
                loss = loss_recon(denoised, clean) * cfg.loss_weights.reconstruction
                if cfg.run_params.use_gradient_loss: loss += cfg.loss_weights.gradient * loss_grad(denoised, clean)
                if cfg.run_params.use_fft_loss: loss += cfg.loss_weights.fft * loss_fft(denoised, clean)
            
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loop.set_postfix(loss=loss.item())
        
        torch.save(model.state_dict(), cfg.paths.save_path)

def train_ecg_classifier(cfg: DictConfig):
    print("--- Training ECG Classifier ---")
    print(OmegaConf.to_yaml(cfg))
    
    beats, labels = load_all_beats_from_dataset(cfg.paths.data_dir)
    if len(beats) == 0: raise RuntimeError("Failed to extract beats.")

    X_train, _, y_train, _ = train_test_split(beats, labels, test_size=0.2, random_state=42, stratify=labels)
    train_dataset = ECGBeatDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training_params.batch_size, shuffle=True)

    model = ECGClassifier(num_classes=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training_params.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.training_params.epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training_params.epochs}")
        for signals, targets in loop:
            signals, targets = signals.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(); outputs = model(signals); loss = criterion(outputs, targets)
            loss.backward(); optimizer.step(); loop.set_postfix(loss=loss.item())
    
    os.makedirs(os.path.dirname(cfg.paths.save_path), exist_ok=True)
    torch.save(model.state_dict(), cfg.paths.save_path)
    print(f"Classifier model saved to {cfg.paths.save_path}")

def train_eeg_model(cfg: DictConfig):
    print(f"--- Starting EEG Training: {cfg.experiment_name} ---")
    print(OmegaConf.to_yaml(cfg))
    
    subject_dir = os.path.join(cfg.paths.data_dir, 'chb01')
    common_channels = find_common_monopolar_channels(subject_dir)
    NUM_CHANNELS = len(common_channels)
    
    all_segments = []
    for f in tqdm(os.listdir(subject_dir), desc="Loading files"):
        if f.endswith('.edf'):
            data, ch_names, _ = load_eeg_from_edf(os.path.join(subject_dir, f))
            if data is not None and set(common_channels).issubset(set(ch_names)):
                ch_map = {name: i for i, name in enumerate(ch_names)}
                ordered_data = data[[ch_map[name] for name in common_channels], :]
                all_segments.extend(create_eeg_segments(ordered_data, cfg.data_params.window_samples))
            
    if cfg.run_params.eeg_experiment_type == 'self_supervised':
        train_dataset = MaskedEEGDataset(all_segments, cfg.training_params.samples_per_epoch)
    else:
        train_dataset = EEGDataset(all_segments, cfg.training_params.samples_per_epoch)

    os.makedirs(os.path.dirname(cfg.paths.save_path), exist_ok=True)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.training_params.batch_size, num_workers=2, pin_memory=True, shuffle=True)
    model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training_params.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))
    
    loss_fns = {'L1': nn.L1Loss()}
    if cfg.run_params.eeg_experiment_type in ['spatial', 'self_supervised']:
        loss_fns['Spatial'] = LaplacianLoss(get_adjacency_list(common_channels))
    # ... other EEG loss logic ...
            
    for epoch in range(cfg.training_params.epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training_params.epochs}")
        for data_in, data_clean in loop:
            data_in, data_clean = data_in.to(DEVICE), data_clean.to(DEVICE)
            with torch.autocast(device_type=DEVICE, enabled=(DEVICE=='cuda')):
                reconstructed = model(data_in)
                loss = loss_fns['L1'](reconstructed, data_clean)
                # ... other EEG loss calculation logic based on cfg ...
            
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loop.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), cfg.paths.save_path)
    print(f"✅ Training complete. Model saved to {cfg.paths.save_path}")

# --- Hydra Entry Point ---
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment_cfg = cfg.experiment
    
    if "ecg_denoiser" in experiment_cfg.experiment_name:
        train_ecg_denoiser(experiment_cfg)
    elif "ecg_classifier" in experiment_cfg.experiment_name:
        train_ecg_classifier(experiment_cfg)
    elif "eeg" in experiment_cfg.experiment_name:
        train_eeg_model(experiment_cfg)
    else:
        raise ValueError(f"Unknown experiment: {experiment_cfg.experiment_name}")

if __name__ == "__main__":
    main()