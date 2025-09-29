# src/stpc/utils/eeg_utils.py
import mne
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

TARGET_FS = 256

def find_common_monopolar_channels(directory):
    """
    Scans all .edf files in a directory to find the intersection of
    standard monopolar channels that are present in every file.
    This is the definitive function for ensuring channel consistency.
    """
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
        except Exception:
            pass
    
    final_common_channels = [standard_channels_map[ch_upper] for ch_upper in common_channels_set]
    return sorted(final_common_channels)

def load_eeg_from_edf(file_path, target_fs=TARGET_FS):
    # ... (This function remains correct)
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
                ch_name_mapping[ch_name] = standard_channels_map[mono_name_upper]
                final_mono_names_upper.append(mono_name_upper)
        raw.pick(list(ch_name_mapping.keys()))
        raw.rename_channels(ch_name_mapping)
        raw.set_eeg_reference('average', projection=False)
        raw.filter(l_freq=0.5, h_freq=70.0)
        raw.notch_filter(freqs=60.0)
        raw.resample(sfreq=target_fs)
        raw.reorder_channels(sorted(raw.ch_names))
        return raw.get_data().astype(np.float32), raw.ch_names, raw.info
    except Exception:
        return None, None, None

def get_adjacency_list(channel_names):
    # ... (This function remains correct)
    montage = mne.channels.make_standard_montage('standard_1020')
    positions, present_channels = [], []
    for ch in channel_names:
        if ch in montage.ch_names:
            positions.append(montage.get_positions()['ch_pos'][ch])
            present_channels.append(ch)
    positions = np.array(positions)
    dist_matrix = squareform(pdist(positions))
    adjacency_matrix = dist_matrix < 0.055
    np.fill_diagonal(adjacency_matrix, False)
    adj_list = [np.where(row)[0].tolist() for row in adjacency_matrix]
    final_adj_list = [[] for _ in channel_names]
    name_to_idx_map = {name: i for i, name in enumerate(channel_names)}
    for i, ch_name in enumerate(present_channels):
        original_idx = name_to_idx_map[ch_name]
        neighbor_names = [present_channels[j] for j in adj_list[i]]
        final_neighbor_indices = [name_to_idx_map[n] for n in neighbor_names]
        final_adj_list[original_idx] = final_neighbor_indices
    return final_adj_list

def create_eeg_segments(data, window_samples, overlap_samples=0):
    # ... (This function remains correct)
    if data is None: return []
    _, n_samples = data.shape
    segments = []
    step = window_samples - overlap_samples
    for start in range(0, n_samples - window_samples + 1, step):
        segments.append(data[:, start:(start + window_samples)])
    return segments