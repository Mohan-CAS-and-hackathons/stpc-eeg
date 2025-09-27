# src/eeg_data_utils.py
import mne
import numpy as np
import os
import torch
from torch.utils.data import Dataset

TARGET_FS = 256

def load_eeg_from_edf(file_path, target_fs=TARGET_FS):
    """
    Loads an EDF, robustly extracts a set of standard monopolar channels,
    and returns the data array and the list of final channel names.
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        montage = mne.channels.make_standard_montage('standard_1020')
        standard_channels_map = {ch.upper(): ch for ch in montage.ch_names}
        standard_channels_upper = set(standard_channels_map.keys())

        ch_name_mapping = {}
        final_mono_names_upper = []
        
        for ch_name in raw.ch_names:
            mono_name_upper = ch_name.split('-')[0].upper()
            if mono_name_upper in standard_channels_upper and mono_name_upper not in final_mono_names_upper:
                # Use the official mixed-case name for the mapping
                ch_name_mapping[ch_name] = standard_channels_map[mono_name_upper]
                final_mono_names_upper.append(mono_name_upper)

        raw.pick(list(ch_name_mapping.keys()))
        raw.rename_channels(ch_name_mapping)

        raw.set_eeg_reference('average', projection=False)
        raw.filter(l_freq=0.5, h_freq=70.0)
        raw.notch_filter(freqs=60.0)
        raw.resample(sfreq=target_fs)
        
        raw.reorder_channels(sorted(raw.ch_names))
        
        return raw.get_data().astype(np.float32), raw.ch_names

    except Exception:
        return None, None

def get_adjacency_list(channel_names):
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=channel_names, sfreq=TARGET_FS, ch_types='eeg')
    
    # --- DEFINITIVE FIX: Use 'ignore' to handle minor discrepancies ---
    info.set_montage(montage, on_missing='ignore')
    # MNE can find neighbors automatically from the montage
    adj, _ = mne.channels.find_ch_connectivity(info, ch_type='eeg')
    adj_matrix = adj.toarray()
    
    adj_list = []
    for i in range(len(channel_names)):
        neighbor_indices = np.where(adj_matrix[i] == 1)[0].tolist()
        adj_list.append(neighbor_indices)
        
    return adj_list

# ... The rest of the file (create_eeg_segments, EEGDataset) remains the same ...
def create_eeg_segments(data, window_samples, overlap_samples=0):
    if data is None: return []
    n_channels, n_samples = data.shape
    segments = []
    step = window_samples - overlap_samples
    for start in range(0, n_samples - window_samples + 1, step):
        segments.append(data[:, start:(start + window_samples)])
    return segments

class EEGDataset(Dataset):
    def __init__(self, clean_segments, samples_per_epoch, noise_level=0.5):
        self.clean_segments = clean_segments
        self.samples_per_epoch = samples_per_epoch
        self.noise_level = noise_level
        cat_data = np.concatenate(self.clean_segments[:min(500, len(self.clean_segments))], axis=1)
        self.mean = np.mean(cat_data, axis=1, keepdims=True).astype(np.float32)
        self.std = np.std(cat_data, axis=1, keepdims=True).astype(np.float32)

    def __len__(self): return self.samples_per_epoch
    def __getitem__(self, idx):
        clean_segment = self.clean_segments[np.random.randint(0, len(self.clean_segments))]
        clean_segment = (clean_segment - self.mean) / (self.std + 1e-8)
        signal_power = np.mean(clean_segment ** 2)
        noise = np.random.randn(*clean_segment.shape).astype(np.float32)
        noise_power = np.mean(noise ** 2) if np.mean(noise ** 2) > 0 else 1e-8
        scale_factor = np.sqrt(signal_power * self.noise_level / noise_power)
        noise_scaled = noise * scale_factor
        return torch.from_numpy(clean_segment + noise_scaled), torch.from_numpy(clean_segment)