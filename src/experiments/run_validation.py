# src/experiments/run_validation.py
import os
import sys
import argparse
import torch
import numpy as np
import wfdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Add project root to system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import from our STPC library, including the new helper ---
from stpc.model import UNet1D, ECGClassifier
from stpc.utils.ecg_utils import (
    TARGET_FS, BEAT_CLASSES, BEAT_WINDOW_SIZE, get_noise_signals,
    load_and_resample_signal, working_directory
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def validate_ecg_downstream(args):
    print("--- Running End-to-End ECG Downstream Validation ---")
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    print("Loading models...")
    denoiser = UNet1D().to(DEVICE); denoiser.load_state_dict(torch.load(args.denoiser_path, map_location=DEVICE)); denoiser.eval()
    classifier = ECGClassifier().to(DEVICE); classifier.load_state_dict(torch.load(args.classifier_path, map_location=DEVICE)); classifier.eval()

    # --- Use the working_directory context manager for all wfdb calls ---
    print(f"Loading record: {args.record_name} from {args.data_dir}")
    with working_directory(args.data_dir):
        clean_signal = load_and_resample_signal(args.record_name, TARGET_FS)
        if clean_signal is None:
            raise RuntimeError(f"Failed to load validation record: {args.record_name}")
        annotation = wfdb.rdann(args.record_name, 'atr')
    
    true_samples = (annotation.sample * (TARGET_FS / annotation.fs)).astype('int64')

    # Synthesize Noise
    noise_signals = get_noise_signals(args.noise_dir, TARGET_FS)
    noise_type = 'muscle_artifact'
    long_noise = np.tile(noise_signals[noise_type], int(np.ceil(len(clean_signal) / len(noise_signals[noise_type]))))[:len(clean_signal)]
    power_clean = np.mean(clean_signal ** 2)
    power_noise = np.mean(long_noise ** 2)
    scaling_factor = np.sqrt((power_clean / (10**(args.snr_db / 10))) / power_noise) if power_noise > 0 else 0
    noisy_signal = clean_signal + long_noise * scaling_factor

    # Denoise Signal
    denoised_signal = np.zeros_like(noisy_signal)
    for i in tqdm(range(0, len(noisy_signal), 2048), desc="Denoising"):
        segment = noisy_signal[i:i+2048]
        if len(segment) < 2048: segment = np.pad(segment, (0, 2048 - len(segment)))
        with torch.no_grad():
            tensor_in = torch.from_numpy(segment.copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            tensor_out = denoiser(tensor_in).squeeze().cpu().numpy()
        denoised_signal[i:i+len(noisy_signal[i:i+2048])] = tensor_out[:len(noisy_signal[i:i+2048])]
        
    # Classify and Evaluate
    signals_to_test = {'Noisy': noisy_signal, 'Denoised': denoised_signal, 'Clean': clean_signal}
    final_class_names = ['N', 'S', 'V', 'F', 'Q']
    
    for name, sig in signals_to_test.items():
        predictions, ground_truth = [], []
        for i, sym in enumerate(annotation.symbol):
            if sym in BEAT_CLASSES:
                loc = true_samples[i]
                start, end = loc - BEAT_WINDOW_SIZE//2, loc + BEAT_WINDOW_SIZE//2
                if start >= 0 and end < len(sig):
                    with torch.no_grad():
                        tensor_in = torch.from_numpy(sig[start:end].copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                        pred_label = torch.argmax(classifier(tensor_in), dim=1).item()
                    predictions.append(pred_label)
                    ground_truth.append(BEAT_CLASSES[sym])
        
        print(f"\n--- PERFORMANCE ON {name.upper()} SIGNAL ---")
        print(classification_report(ground_truth, predictions, target_names=final_class_names, zero_division=0))
        cm = confusion_matrix(ground_truth, predictions, labels=range(len(final_class_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_class_names)
        disp.plot(cmap=plt.cm.Blues); disp.ax_.set_title(f'Confusion Matrix - {name} Signal')
        plt.savefig(f'{args.output_prefix}_cm_{name.lower()}.png', bbox_inches='tight'); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Validation Runner for STPC Experiments")
    subparsers = parser.add_subparsers(dest="validation_task", required=True)

    p_ecg = subparsers.add_parser("ecg_downstream")
    p_ecg.add_argument("--denoiser_path", type=str, required=True)
    p_ecg.add_argument("--classifier_path", type=str, required=True)
    p_ecg.add_argument("--data_dir", type=str, required=True)
    p_ecg.add_argument("--noise_dir", type=str, required=True)
    p_ecg.add_argument("--output_prefix", type=str, required=True)
    p_ecg.add_argument("--record_name", type=str, default="201")
    p_ecg.add_argument("--snr_db", type=int, default=0)
    
    args = parser.parse_args()

    if args.validation_task == "ecg_downstream":
        validate_ecg_downstream(args)