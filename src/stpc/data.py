# src/stpc/data.py
import torch
from torch.utils.data import Dataset
import numpy as np
import random

from .utils.ecg_utils import create_noisy_clean_pair # Note the new import path

# --- ECG Datasets ---

class PhysioNetDataset(Dataset):
    """
    A PyTorch dataset that generates noisy-clean ECG pairs on the fly.
    """
    def __init__(self, clean_signals, noise_signals, train_params: dict, num_samples_per_epoch: int):
        self.clean_signals = clean_signals
        self.noise_signals = noise_signals
        self.train_params = train_params
        self.num_samples = num_samples_per_epoch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clean_signal = random.choice(self.clean_signals)
        snr_db = random.uniform(self.train_params['snr_db_min'], self.train_params['snr_db_max'])
        
        noisy_segment, clean_segment = None, None
        while noisy_segment is None: # Retry if generation fails
            noisy_segment, clean_segment = create_noisy_clean_pair(
                clean_signal=clean_signal,
                noise_signals=self.noise_signals,
                segment_samples=self.train_params['segment_samples'],
                snr_db=snr_db
            )

        noisy_tensor = torch.from_numpy(noisy_segment.copy()).float().unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_segment.copy()).float().unsqueeze(0)

        return noisy_tensor, clean_tensor

class ECGBeatDataset(Dataset):
    """PyTorch Dataset for ECG beat classification."""
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal_tensor = torch.from_numpy(self.signals[idx]).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal_tensor, label_tensor

# --- EEG Datasets ---

class EEGDataset(Dataset):
    def __init__(self, clean_segments, samples_per_epoch, noise_level=0.5):
        self.clean_segments = clean_segments
        self.samples_per_epoch = samples_per_epoch
        self.noise_level = noise_level
        # Calculate normalization stats from a subset of the data
        cat_data = np.concatenate(self.clean_segments[:min(500, len(self.clean_segments))], axis=1)
        self.mean = np.mean(cat_data, axis=1, keepdims=True).astype(np.float32)
        self.std = np.std(cat_data, axis=1, keepdims=True).astype(np.float32)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        clean_segment = self.clean_segments[np.random.randint(0, len(self.clean_segments))]
        # Normalize
        clean_segment_norm = (clean_segment - self.mean) / (self.std + 1e-8)
        
        # Create noise
        signal_power = np.mean(clean_segment_norm ** 2)
        noise = np.random.randn(*clean_segment_norm.shape).astype(np.float32)
        noise_power = np.mean(noise ** 2) if np.mean(noise ** 2) > 0 else 1e-8
        scale_factor = np.sqrt(signal_power * self.noise_level / noise_power)
        noise_scaled = noise * scale_factor
        
        noisy_segment = clean_segment_norm + noise_scaled
        
        return torch.from_numpy(noisy_segment), torch.from_numpy(clean_segment_norm)

class MaskedEEGDataset(EEGDataset):
    """
    Inherits from EEGDataset but returns a masked version of the clean signal
    as the input 'x', with the original clean signal as the target 'y'.
    """
    def __init__(self, clean_segments, samples_per_epoch, mask_ratio=0.4):
        super().__init__(clean_segments, samples_per_epoch)
        self.mask_ratio = mask_ratio

    def __getitem__(self, idx):
        # Get a normalized clean segment from the parent class logic
        _, clean_segment_norm_tensor = super().__getitem__(idx)
        
        # Create a random mask
        T = clean_segment_norm_tensor.shape[1]
        num_masked = int(self.mask_ratio * T)
        masked_indices = np.random.choice(T, num_masked, replace=False)
        
        masked_segment = clean_segment_norm_tensor.clone()
        masked_segment[:, masked_indices] = 0 # Masking value is 0
        
        return masked_segment, clean_segment_norm_tensor