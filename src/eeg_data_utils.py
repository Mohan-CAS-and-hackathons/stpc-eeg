# src/eeg_data_utils.py
import mne
import numpy as np
import os
import torch
from torch.utils.data import Dataset

# This file now contains the definitive, robust data loading logic.

TARGET_FS = 256

def load_eeg_from_edf(file_path, target_fs=TARGET_FS):
    """
    Loads an EDF, robustly extracts a set of standard monopolar channels,
    and returns the data array and the list of final channel names.
    This is the definitive data loading function.
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # 1. Get a standard 10-20 montage from MNE
        montage = mne.channels.make_standard_montage('standard_1020')
        standard_channels_upper = {ch.upper() for ch in montage.ch_names}

        # 2. Create a mapping from the file's original channel names to standard monopolar names
        ch_name_mapping = {}
        final_monopolar_names = []
        
        for ch_name in raw.ch_names:
            # Extract the first part of the bipolar name (e.g., 'FP1' from 'FP1-F7')
            mono_name = ch_name.split('-')[0].upper()
            
            # Check if this monopolar name is in the standard montage AND we haven't added it yet
            if mono_name in standard_channels_upper and mono_name not in final_monopolar_names:
                ch_name_mapping[ch_name] = mono_name
                final_monopolar_names.append(mono_name)
        
        # 3. Pick only the channels we could successfully map and rename them
        raw.pick(list(ch_name_mapping.keys()))
        raw.rename_channels(ch_name_mapping)

        # 4. Standard preprocessing
        raw.set_eeg_reference('average', projection=False)
        raw.filter(l_freq=0.5, h_freq=70.0)
        raw.notch_filter(freqs=60.0)
        raw.resample(sfreq=target_fs)
        
        # 5. Reorder channels alphabetically to ensure consistency across all files
        raw.reorder_channels(sorted(raw.ch_names))
        
        data = raw.get_data()
        return data.astype(np.float32), raw.ch_names

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

def get_adjacency_list(channel_names):
    # This function is now much simpler as it operates on clean, monopolar names
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=channel_names, sfreq=TARGET_FS, ch_types='eeg')
    info.set_montage(montage, on_missing='raise') # Should not have missing channels now
    
    # MNE can find neighbors automatically from the montage
    adj, names = mne.channels.find_ch_connectivity(info, ch_type='eeg')
    adj_matrix = adj.toarray()
    
    adj_list = []
    for i in range(len(channel_names)):
        # Find indices of neighbors from the adjacency matrix
        neighbor_indices = np.where(adj_matrix[i] == 1)[0].tolist()
        adj_list.append(neighbor_indices)
        
    return adj_list

# ... The rest of the file (create_eeg_segments, EEGDataset) remains the same ...
def create_eeg_segments(data, window_samples, overlap_samples=0):
    if data is None:
        return []
    n_channels, n_samples = data.shape
    segments = []
    step = window_samples - overlap_samples
    for start in range(0, n_samples - window_samples + 1, step):
        end = start + window_samples
        segments.append(data[:, start:end])
    return segments

class EEGDataset(Dataset):
    def __init__(self, clean_segments, samples_per_epoch, noise_level=0.5):
        self.clean_segments = clean_segments
        self.samples_per_epoch = samples_per_epoch
        self.noise_level = noise_level
        cat_data = np.concatenate(self.clean_segments[:min(500, len(self.clean_segments))], axis=1)
        self.mean = np.mean(cat_data, axis=1, keepdims=True).astype(np.float32)
        self.std = np.std(cat_data, axis=1, keepdims=True).astype(np.float32)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        clean_segment_idx = np.random.randint(0, len(self.clean_segments))
        clean_segment = self.clean_segments[clean_segment_idx]
        clean_segment = (clean_segment - self.mean) / (self.std + 1e-8)
        signal_power = np.mean(clean_segment ** 2)
        noise = np.random.randn(*clean_segment.shape).astype(np.float32)
        noise_power = np.mean(noise ** 2) if np.mean(noise ** 2) > 0 else 1e-8
        scale_factor = np.sqrt(signal_power * self.noise_level / noise_power)
        noise_scaled = noise * scale_factor
        noisy_segment = clean_segment + noise_scaled
        return torch.from_numpy(noisy_segment), torch.from_numpy(clean_segment)