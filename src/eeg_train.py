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
from eeg_data_utils import load_eeg_from_edf, create_eeg_segments, EEGDataset, TARGET_FS, get_adjacency_list

def find_common_monopolar_channels(directory):
    common_channels_set = None
    montage = mne.channels.make_standard_montage('standard_1020')
    standard_channels_map = {ch.upper(): ch for ch in montage.ch_names}
    standard_channels_upper = set(standard_channels_map.keys())
    
    files_to_check = [f for f in os.listdir(directory) if f.endswith('.edf')]
    for f in tqdm(files_to_check, desc="Scanning for common channels"):
        file_path = os.path.join(directory, f)
        try:
            raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            current_monopolar_upper = set()
            for ch_name in raw.ch_names:
                mono_name_upper = ch_name.split('-')[0].upper()
                if mono_name_upper in standard_channels_upper:
                    current_monopolar_upper.add(mono_name_upper)
            if common_channels_set is None:
                common_channels_set = current_monopolar_upper
            else:
                common_channels_set.intersection_update(current_monopolar_upper)
        except Exception: pass
    
    final_common_channels = [standard_channels_map[ch_upper] for ch_upper in common_channels_set]
    return sorted(final_common_channels)

class TemporalGradientLoss(nn.Module):
    def __init__(self): super().__init__(); self.loss = nn.L1Loss()
    def forward(self, pred, target): return self.loss(torch.diff(pred, dim=-1), torch.diff(target, dim=-1))

class LaplacianLoss(nn.Module):
    def __init__(self, adj_list): super().__init__(); self.adj_list = adj_list; self.loss = nn.L1Loss()
    def _calculate_laplacian(self, x):
        L = torch.zeros_like(x)
        for i, neighbors in enumerate(self.adj_list):
            if len(neighbors) > 0: L[:, i, :] = x[:, i, :] - torch.mean(x[:, neighbors, :], dim=1)
        return L
    def forward(self, pred, target): return self.loss(self._calculate_laplacian(pred), self._calculate_laplacian(target))

class BandMaskedFFTLoss(nn.Module):
    def __init__(self, fs, f_low, f_high):
        super(BandMaskedFFTLoss, self).__init__()
        self.fs, self.f_low, self.f_high = fs, f_low, f_high
        self.loss = nn.L1Loss(); self.mask = None
    def forward(self, pred, target):
        if self.mask is None:
            T = pred.shape[-1]
            freqs = torch.fft.rfftfreq(T, 1.0 / self.fs).to(pred.device)
            self.mask = ((freqs >= self.f_low) & (freqs <= self.f_high)).float()[None, None, :]
        mag_p = torch.abs(torch.fft.rfft(pred, dim=-1))
        mag_c = torch.abs(torch.fft.rfft(target, dim=-1))
        return self.loss(mag_p * self.mask, mag_c * self.mask)

class Config:
    DEFAULT_DATA_DIR = "data/chb-mit-scalp-eeg-database-1.0.0/"
    DEFAULT_MODEL_SAVE_PATH = "models/eeg_denoiser_base.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    NUM_WORKERS = 2
    WINDOW_SECONDS = 2
    WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_FS)
    SAMPLES_PER_EPOCH = 5000

def train_one_epoch(loader, model, optimizer, loss_fns, scaler, config, args):
    loop = tqdm(loader, leave=True)
    total_loss = 0.0
    for noisy, clean in loop:
        noisy, clean = noisy.to(config.DEVICE), clean.to(config.DEVICE)
        with torch.autocast(device_type=config.DEVICE, dtype=torch.float16, enabled=config.DEVICE=='cuda'):
            denoised = model(noisy)
            loss = loss_fns['L1'](denoised, clean)
            if args.experiment == 'spatial':
                loss += (args.alpha * loss_fns['Temporal'](denoised, clean)) + \
                        (args.beta * loss_fns['Spatial'](denoised, clean))
            elif args.experiment == 'frequency':
                loss += (5.0 * loss_fns['Alpha'](denoised, clean)) + \
                        (0.1 * loss_fns['Low'](denoised, clean)) + \
                        (0.1 * loss_fns['High'](denoised, clean))
        if config.DEVICE == 'cuda':
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def main(args):
    config = Config()
    print(f"--- Starting EEG Denoiser Training for experiment: {args.experiment} ---")
    
    subject_dir = os.path.join(args.data_dir, 'chb01')
    common_channels = find_common_monopolar_channels(subject_dir)
    config.NUM_CHANNELS = len(common_channels)
    print(f"Found {config.NUM_CHANNELS} common monopolar channels.")
    
    training_segments = []
    
    for f in tqdm(os.listdir(subject_dir), desc="Loading and processing files"):
        if f.endswith('.edf'):
            if 'chb01_03.edf' in f:
                print(f"\nExcluding {f} from training data (reserved for spatial validation).")
                continue
            
            file_path = os.path.join(subject_dir, f)
            data, ch_names = load_eeg_from_edf(file_path)
            
            if data is not None and ch_names is not None and set(ch_names) == set(common_channels):
                ch_map = {name: i for i, name in enumerate(ch_names)}
                ordered_indices = [ch_map[name] for name in common_channels]
                ordered_data = data[ordered_indices, :]
                segments = create_eeg_segments(ordered_data, config.WINDOW_SAMPLES)
                training_segments.extend(segments)
            
    if not training_segments: raise RuntimeError("Total valid segments for training is 0.")
    print(f"Total valid segments for training: {len(training_segments)}")
    
    train_dataset = EEGDataset(clean_segments=training_segments, samples_per_epoch=config.SAMPLES_PER_EPOCH)
    
    stats_dir = os.path.dirname(args.model_save_path)
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "norm_stats.npz")
    np.savez(stats_path, mean=train_dataset.mean, std=train_dataset.std, channels=common_channels)
    print(f"Normalization stats saved to {stats_path}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                              pin_memory=(config.DEVICE=='cuda'), shuffle=True)
    
    model = UNet1D(in_channels=config.NUM_CHANNELS, out_channels=config.NUM_CHANNELS).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE=='cuda'))
    
    loss_functions = {'L1': nn.L1Loss()}
    if args.experiment == 'spatial':
        adjacency_list = get_adjacency_list(common_channels)
        loss_functions['Temporal'] = TemporalGradientLoss()
        loss_functions['Spatial'] = LaplacianLoss(adjacency_list)
    elif args.experiment == 'frequency':
        loss_functions['Alpha'] = BandMaskedFFTLoss(fs=TARGET_FS, f_low=8, f_high=12)
        loss_functions['Low'] = BandMaskedFFTLoss(fs=TARGET_FS, f_low=0.5, f_high=8)
        loss_functions['High'] = BandMaskedFFTLoss(fs=TARGET_FS, f_low=12, f_high=70)

    for epoch in range(config.NUM_EPOCHS):
        avg_loss = train_one_epoch(train_loader, model, optimizer, loss_functions, scaler, config, args)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Average Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), args.model_save_path)

    print(f"\n✅ Training complete. Model saved to {args.model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='baseline',
                        choices=['baseline', 'spatial', 'frequency', 'self_supervised'])
    parser.add_argument('--data_dir', type=str, default=Config.DEFAULT_DATA_DIR)
    parser.add_argument('--model_save_path', type=str, default=Config.DEFAULT_MODEL_SAVE_PATH)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    cli_args = parser.parse_args()
    main(cli_args)