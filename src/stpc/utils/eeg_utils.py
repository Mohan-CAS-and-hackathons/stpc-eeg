# src/stpc/utils/eeg_utils.py
import mne
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

TARGET_FS = 256

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

def load_eeg_from_edf(file_path, target_fs=TARGET_FS):
    """
    Definitively loads an EDF, processes it to a consistent monopolar format,
    and returns the data, channel names, and a fully configured MNE info object.
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
                ch_name_mapping[ch_name] = standard_channels_map[mono_name_upper]
                final_mono_names_upper.append(mono_name_upper)

        raw.pick(list(ch_name_mapping.keys()))
        raw.rename_channels(ch_name_mapping)
        
        # --- DEFINITIVE FIX ---
        # Set the montage AFTER all channel renaming and selection is complete.
        # This ensures the 3D locations are correctly associated with the final channel names.
        raw.set_montage(montage, on_missing='ignore')
        # --- END FIX ---

        raw.set_eeg_reference('average', projection=False)
        raw.filter(l_freq=0.5, h_freq=70.0)
        raw.notch_filter(freqs=60.0)
        raw.resample(sfreq=target_fs)
        raw.reorder_channels(sorted(raw.ch_names))
        
        return raw.get_data().astype(np.float32), raw.ch_names, raw.info

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, None

def get_adjacency_list(channel_names):
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=channel_names, sfreq=TARGET_FS, ch_types='eeg')
    info.set_montage(montage, on_missing='raise')
    adj, _ = mne.channels.find_ch_connectivity(info, ch_type='eeg')
    adj_matrix = adj.toarray()
    return [np.where(row)[0].tolist() for row in adj_matrix]

def create_eeg_segments(data, window_samples, overlap_samples=0):
    if data is None: return []
    _, n_samples = data.shape
    segments = []
    step = window_samples - overlap_samples
    for start in range(0, n_samples - window_samples + 1, step):
        segments.append(data[:, start:(start + window_samples)])
    return segments