# src/eeg_data_utils.py
import mne
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# --- Configuration ---
# MODIFICATION: This is a robust list of 21 channels commonly found across the CHB-MIT dataset.
# We removed the channels that were causing errors ('PZ-OZ') and others that are sometimes absent.
# This list is more likely to be present in all files.
CHB_MIT_CHANNELS_ROBUST = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
    # Extra common channels
    'P7-T7', 'T7-FT9', 'FT9-FT10'
]
TARGET_FS = 256  # Target sampling frequency for all EEG data

def load_eeg_from_edf(file_path, target_fs=TARGET_FS, desired_channels=CHB_MIT_CHANNELS_ROBUST):
    """
    Loads an EEG signal from an .edf file, preprocesses it, and returns a NumPy array.
    This version is robust to missing channels.
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # --- KEY MODIFICATION: Robust channel selection ---
        # Instead of failing, we find which of our desired channels are actually available.
        available_channels = [ch.upper() for ch in raw.ch_names]
        channels_to_pick = [ch for ch in desired_channels if ch.upper() in available_channels]

        # If a file has too few of our desired channels, we can skip it.
        if len(channels_to_pick) < 18: # A reasonable threshold
            print(f"Warning: Skipping file {os.path.basename(file_path)} - found only {len(channels_to_pick)} of desired channels.")
            return None
            
        # Use the modern .pick() method with the list of channels that are guaranteed to exist.
        raw.pick(channels_to_pick)
        # ---------------------------------------------------

        raw.set_eeg_reference('average', projection=False)
        raw.filter(l_freq=0.5, h_freq=70.0)
        raw.notch_filter(freqs=60.0)
        raw.resample(sfreq=target_fs)
        data = raw.get_data()
        return data.astype(np.float32)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# ... (The rest of the file remains exactly the same)

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

if __name__ == '__main__':
    DATA_DIR = "data/chb-mit-scalp-eeg-database-1.0.0/"
    SAMPLE_FILE = os.path.join(DATA_DIR, "chb01/chb01_03.edf")
    
    if not os.path.exists(SAMPLE_FILE):
        print(f"Error: Sample file not found at {SAMPLE_FILE}")
        print("Please ensure your data is correctly placed.")
    else:
        print(f"--- Testing EEG Data Utilities with {SAMPLE_FILE} ---")
        
        print("1. Loading and preprocessing data...")
        preprocessed_data = load_eeg_from_edf(SAMPLE_FILE)
        assert preprocessed_data is not None, "Data loading failed."
        print(f"   Success! Data shape: {preprocessed_data.shape} (Channels, Samples)")

        print("\n2. Splitting data into 2-second segments...")
        WINDOW_SAMPLES = 2 * TARGET_FS
        segments = create_eeg_segments(preprocessed_data, WINDOW_SAMPLES)
        assert len(segments) > 0, "Segmenting failed."
        print(f"   Success! Created {len(segments)} segments of shape {segments[0].shape}")
        
        print("\n3. Initializing EEGDataset...")
        dataset = EEGDataset(clean_segments=segments, samples_per_epoch=100)
        assert len(dataset) == 100, "Dataset initialization failed."
        noisy, clean = dataset[0]
        print(f"   Success! Dataset returned noisy/clean pair with shape: {noisy.shape}")
        
        print("\n✅ Phase 0 Data Utilities are now robust and working correctly!")