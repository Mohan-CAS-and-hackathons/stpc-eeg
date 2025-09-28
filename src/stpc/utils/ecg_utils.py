# src/stpc/utils/ecg_utils.py
import wfdb
import numpy as np
import os
from scipy.signal import resample
from tqdm import tqdm
from collections import Counter

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

# --- Data Loading and Synthesis Functions (from data_utils.py) ---

def get_all_record_names(data_path):
    all_files = os.listdir(data_path)
    record_names = sorted(list(set([f.split('.')[0] for f in all_files])))
    return [name for name in record_names if name.isdigit()]

def load_and_resample_signal(record_path, target_fs):
    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]
        original_fs = record.fs

        if original_fs == target_fs:
            return signal
        
        num_samples_target = int(len(signal) * target_fs / original_fs)
        resampled_signal = resample(signal, num_samples_target)
        return resampled_signal
    except Exception as e:
        print(f"Warning: Could not process record {record_path}. Error: {e}")
        return None

def get_noise_signals(noise_data_path, target_fs):
    noise_types = {'bw': 'baseline_wander', 'em': 'electrode_motion', 'ma': 'muscle_artifact'}
    noises = {}
    for noise_code in noise_types:
        record_path = os.path.join(noise_data_path, noise_code)
        noise_signal = load_and_resample_signal(record_path, target_fs)
        if noise_signal is not None:
            noises[noise_types[noise_code]] = noise_signal
    return noises

def create_noisy_clean_pair(clean_signal, noise_signals, segment_samples, snr_db):
    if len(clean_signal) < segment_samples: return None, None
    
    start_index = np.random.randint(0, len(clean_signal) - segment_samples)
    clean_segment = clean_signal[start_index : start_index + segment_samples]

    noise_type = np.random.choice(list(noise_signals.keys()))
    noise_signal = noise_signals[noise_type]
    
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

# --- Beat Extraction Functions (from classification_data.py) ---

def load_all_beats_from_dataset(data_path):
    all_beats, all_labels = [], []
    record_names = get_all_record_names(data_path)
    
    print("Extracting all annotated heartbeats from the dataset...")
    for rec_name in tqdm(sorted(record_names)):
        try:
            record_path = os.path.join(data_path, rec_name)
            signal = load_and_resample_signal(record_path, TARGET_FS)
            if signal is None: continue
                
            annotation = wfdb.rdann(record_path, 'atr')
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
        except Exception as e:
            print(f"Warning: Could not process record {rec_name}. Error: {e}")

    print(f"Extracted a total of {len(all_beats)} beats.")
    print(f"Label distribution: {Counter(all_labels)}")
    return np.array(all_beats, dtype=np.float32), np.array(all_labels, dtype=np.int64)