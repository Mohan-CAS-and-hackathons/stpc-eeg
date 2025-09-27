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
    """
    Creates a list of neighbor indices based on 3D sensor locations.
    This is the robust, version-independent method.
    """
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Get the 3D positions of the channels we are using
    positions = []
    present_channels = []
    for ch in channel_names:
        if ch in montage.ch_names:
            ch_index = montage.ch_names.index(ch)
            positions.append(montage.get_positions()['ch_pos'][ch])
            present_channels.append(ch)

    positions = np.array(positions)
    
    # Calculate pairwise distances between all channels
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(positions))
    
    # Define neighbors as channels within a certain radius (e.g., 0.05 meters)
    # This is a common way to define adjacency in MNE
    adjacency_matrix = dist_matrix < 0.05
    # A channel is not its own neighbor
    np.fill_diagonal(adjacency_matrix, False)
    
    adj_list = []
    for i in range(len(present_channels)):
        neighbor_indices = np.where(adjacency_matrix[i])[0].tolist()
        adj_list.append(neighbor_indices)
        
    # The adjacency list must match the order of the original channel_names.
    # We need to map the indices from `present_channels` back to `channel_names`.
    final_adj_list = [[] for _ in channel_names]
    name_to_idx_map = {name: i for i, name in enumerate(channel_names)}
    
    for i, ch_name in enumerate(present_channels):
        original_idx = name_to_idx_map[ch_name]
        neighbor_names = [present_channels[j] for j in adj_list[i]]
        final_neighbor_indices = [name_to_idx_map[n] for n in neighbor_names]
        final_adj_list[original_idx] = final_neighbor_indices

    return final_adj_list

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