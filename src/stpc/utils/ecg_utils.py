# src/stpc/utils/ecg_utils.py
import wfdb
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
from scipy.signal import resample
import contextlib

from typing import Optional

import numpy as np

# --- DEFINITIVE FIX: A context manager to handle directory changes safely ---
@contextlib.contextmanager
def working_directory(path):
    """A context manager to temporarily change the working directory."""
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
# ---

# --- Constants ---
TARGET_FS = 250
BEAT_WINDOW_SIZE = 128
BEAT_CLASSES = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 'n': 0, # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,                # Supraventricular
    'V': 2, 'E': 2,                                # Ventricular
    'F': 3,                                        # Fusion
    'Q': 4, '?': 4, '|': 4, '/': 4,                # Unknown
}

# --- Data Loading Functions ---

def get_all_record_names(data_path):
    try:
        all_files = os.listdir(data_path)
        record_names = sorted(list(set([f.split('.')[0] for f in all_files])))
        return [name for name in record_names if name.isdigit()]
    except FileNotFoundError:
        return []

def load_and_resample_signal(record_name: str, target_fs: int) -> Optional[np.ndarray]:
    """Loads a signal by its name from the CURRENT working directory."""
    try:
        record = wfdb.rdrecord(record_name)
        signal = record.p_signal[:, 0]
        original_fs = record.fs
        if original_fs == target_fs:
            return signal
        num_samples_target = int(len(signal) * target_fs / original_fs)
        resampled_signal = resample(signal, num_samples_target)
        return resampled_signal
    except Exception:
        return None

def get_noise_signals(noise_data_path, target_fs):
    noises = {}
    noise_types = {'bw': 'baseline_wander', 'em': 'electrode_motion', 'ma': 'muscle_artifact'}
    # Use the context manager to ensure noise files are found
    with working_directory(noise_data_path):
        for noise_code in noise_types:
            noise_signal = load_and_resample_signal(noise_code, target_fs)
            if noise_signal is not None:
                noises[noise_types[noise_code]] = noise_signal
    return noises

def create_noisy_clean_pair(clean_signal, noise_signals, segment_samples, snr_db):
    if not noise_signals or len(clean_signal) < segment_samples: return None, None
    start_index = np.random.randint(0, len(clean_signal) - segment_samples)
    clean_segment = clean_signal[start_index : start_index + segment_samples]
    noise_type = np.random.choice(list(noise_signals.keys()))
    noise_signal = noise_signals[noise_type]
    if len(noise_signal) < segment_samples: return None, None
    noise_start_index = np.random.randint(0, len(noise_signal) - segment_samples)
    noise_segment = noise_signal[noise_start_index : noise_start_index + segment_samples]
    power_clean = np.mean(clean_segment ** 2)
    power_noise_initial = np.mean(noise_segment ** 2)
    if power_noise_initial == 0: return None, None
    snr_linear = 10 ** (snr_db / 10)
    target_noise_power = power_clean / snr_linear
    scaling_factor = np.sqrt(target_noise_power / power_noise_initial)
    scaled_noise_segment = noise_segment * scaling_factor
    noisy_segment = clean_segment + scaled_noise_segment
    return noisy_segment.astype(np.float32), clean_segment.astype(np.float32)

def load_all_beats_from_dataset(data_path):
    all_beats, all_labels = [], []
    record_names = get_all_record_names(data_path)
    if not record_names:
        return np.array(all_beats), np.array(all_labels)

    print("Extracting all annotated heartbeats from the dataset...")
    # Use the context manager to safely change into the data directory
    with working_directory(data_path):
        for rec_name in tqdm(record_names):
            # Now that we are IN the directory, we pass the base name
            signal = load_and_resample_signal(rec_name, TARGET_FS)
            if signal is None:
                continue
            try:
                # This call will now succeed because it's in the same folder
                annotation = wfdb.rdann(rec_name, 'atr')
                ann_samples = annotation.sample.astype('int64')
                rescaled_ann_samples = np.round(ann_samples * (TARGET_FS / annotation.fs)).astype(int)

                for i, symbol in enumerate(annotation.symbol):
                    if symbol in BEAT_CLASSES:
                        label = BEAT_CLASSES[symbol]
                        r_peak_loc = rescaled_ann_samples[i]
                        start, end = r_peak_loc - BEAT_WINDOW_SIZE // 2, r_peak_loc + BEAT_WINDOW_SIZE // 2
                        
                        if start >= 0 and end < len(signal):
                            all_beats.append(signal[start:end])
                            all_labels.append(label)
            except Exception:
                pass
                
    print(f"Extracted a total of {len(all_beats)} beats.")
    print(f"Label distribution: {Counter(all_labels)}")
    return np.array(all_beats, dtype=np.float32), np.array(all_labels, dtype=np.int64)