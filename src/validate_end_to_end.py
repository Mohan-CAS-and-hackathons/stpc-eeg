import torch
import numpy as np
import os
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import wfdb

from wfdb.processing import resample_sig

# Imports from our project
from data_utils import load_and_resample_signal, get_noise_signals, TARGET_FS
from classification_data import BEAT_CLASSES, BEAT_WINDOW_SIZE
from model import UNet1D
from classifier_model import ECGClassifier

# --- Configuration ---
DRIVE_PATH = '/content/drive/MyDrive/ecg_denoiser_hackathon/'
# Note: NOISE_DATA_PATH is still needed, but clean data will be downloaded
NOISE_DATA_PATH = os.path.join(DRIVE_PATH, 'data/mit-bih-noise-stress-test-database-1.0.0/')
CLASSIFIER_MODEL_PATH = os.path.join(DRIVE_PATH, 'models/ecg_classifier_model.pth')
DEFAULT_DENOISER_PATH = os.path.join(DRIVE_PATH, 'models/denoiser_stpc_full.pth')
DEFAULT_OUTPUT_PREFIX = os.path.join(DRIVE_PATH, 'results/stpc_full')

VALIDATION_RECORD_NAME = '201'
NOISE_SNR_DB = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    print("--- Starting End-to-End Validation ---")
    print(f"Using Device: {DEVICE}")
    print(f"Test Record: {VALIDATION_RECORD_NAME}, Noise Level: {NOISE_SNR_DB} dB SNR")
    print(f"Loading Denoiser from: {args.denoiser_model_path}")
    print(f"Output file prefix: {args.output_prefix}")

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Loading models...")
    denoiser = UNet1D().to(DEVICE)
    denoiser.load_state_dict(torch.load(args.denoiser_model_path, map_location=DEVICE))
    denoiser.eval()

    classifier = ECGClassifier().to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
    classifier.eval()

    # --- DEFINITIVE FIX: Download record and annotation directly from PhysioNet ---
    print(f"Downloading fresh copy of record '{VALIDATION_RECORD_NAME}' from PhysioNet to bypass local read errors...")
    # This downloads .dat, .hea, and .atr files to the current directory
    wfdb.dl_database('mitdb', '.', records=[VALIDATION_RECORD_NAME])
    print("Download complete.")
    
    # Now, read the freshly downloaded local record
    record = wfdb.rdrecord(VALIDATION_RECORD_NAME)
    annotation = wfdb.rdann(VALIDATION_RECORD_NAME, 'atr')
    
    # Resample the signal to our target frequency
    clean_signal = resample_sig(record.p_signal[:, 0], record.fs, TARGET_FS)[0]
    true_symbols = annotation.symbol
    # Scale annotation samples to the new frequency
    true_samples = (annotation.sample * (TARGET_FS / record.fs)).astype('int64')
    # --------------------------------------------------------------------------

    print("Synthesizing a highly noisy signal...")
    noise_signals = get_noise_signals(NOISE_DATA_PATH, TARGET_FS)
    print("Available noise types:", noise_signals.keys())

    # Map short names to actual keys in your noise_signals dict
    noise_alias = {
        "bw": "baseline_wander",
        "em": "electrode_motion",
        "ma": "muscle_artifact",
    }

    # Select noise type (fallback order)
    for alias in ["bw", "em", "ma"]:
        if noise_alias[alias] in noise_signals:
            noise_type = noise_alias[alias]
            break
    else:
        raise RuntimeError("No valid noise signals available!")

    print(f"Using '{noise_type}' noise for validation.")
    long_noise = np.tile(
        noise_signals[noise_type],
        int(np.ceil(len(clean_signal) / len(noise_signals[noise_type])))
    )

    long_noise = np.tile(noise_signals[noise_type], int(np.ceil(len(clean_signal) / len(noise_signals[noise_type]))))
    long_noise = long_noise[:len(clean_signal)]

    power_clean = np.mean(clean_signal ** 2)
    power_noise = np.mean(long_noise ** 2)
    scaling_factor = np.sqrt((power_clean / (10**(NOISE_SNR_DB / 10))) / power_noise) if power_noise > 0 else 0
    noisy_signal = clean_signal + long_noise * scaling_factor

    print("Denoising the signal with the U-Net model...")
    segment_length = 2048
    denoised_signal = np.zeros_like(noisy_signal)
    for i in range(0, len(noisy_signal), segment_length):
        segment = noisy_signal[i:i+segment_length]
        original_len = len(segment)

        if original_len < segment_length:
            padded_segment = np.zeros(segment_length, dtype=np.float32)
            padded_segment[:original_len] = segment
            segment = padded_segment

        with torch.no_grad():
            tensor_in = torch.from_numpy(segment).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            tensor_out = denoiser(tensor_in).squeeze(0).squeeze(0).cpu().numpy()

        denoised_signal[i:i+segment_length] = tensor_out[:original_len]

    print("Classifying beats from all three signal types...")
    signals_to_test = {'Noisy': noisy_signal, 'Denoised': denoised_signal, 'Clean': clean_signal}
    results = {}

    for name, sig in signals_to_test.items():
        predictions = []
        ground_truth = []

        for i, sym in enumerate(true_symbols):
            if sym in BEAT_CLASSES:
                loc = true_samples[i]
                start, end = loc - BEAT_WINDOW_SIZE//2, loc + BEAT_WINDOW_SIZE//2
                if start >= 0 and end < len(sig):
                    beat_window = sig[start:end]
                    with torch.no_grad():
                        tensor_in = torch.from_numpy(beat_window.copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                        pred_logit = classifier(tensor_in)
                        pred_label = torch.argmax(pred_logit, dim=1).item()

                    predictions.append(pred_label)
                    ground_truth.append(BEAT_CLASSES[sym])

        results[name] = {'preds': predictions, 'truth': ground_truth}

    # Define the FINAL 5 class names for plotting
    final_class_names = ['N', 'S', 'V', 'F', 'Q']
    final_class_labels = range(len(final_class_names))

    for name, data in results.items():
        print(f"\n--- PERFORMANCE ON {name.upper()} SIGNAL ---")
        report = classification_report(
            data['truth'],
            data['preds'],
            target_names=final_class_names, # Use corrected names
            labels=final_class_labels,       # Use corrected labels
            zero_division=0
        )
        print(report)

        cm = confusion_matrix(data['truth'], data['preds'], labels=final_class_labels) # Use corrected labels
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_class_names) # Use corrected names
        disp.plot(cmap=plt.cm.Blues)
        disp.ax_.set_title(f'Confusion Matrix - {name} Signal')

        output_filename = f'{args.output_prefix}_confusion_matrix_{name.lower().replace(" ", "_")}.png'
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_filename}")
        plt.close()

    print("\n✅ Validation complete. Check the classification reports and saved plots.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate a denoiser model end-to-end against a classifier.')
    parser.add_argument('--denoiser_model_path', type=str, default=DEFAULT_DENOISER_PATH, help='Path to the trained denoiser model (.pth) file.')
    parser.add_argument('--output_prefix', type=str, default=DEFAULT_OUTPUT_PREFIX, help='Prefix for saving output files.')

    cli_args = parser.parse_args()
    main(cli_args)