# src/experiments/run_training.py
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import yaml

# ==============================================================================
#                      SETUP: Add project root to system path
# ==============================================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==============================================================================

# --- Import from our STPC library ---
from src.stpc.model import UNet1D, ECGClassifier
from src.stpc.losses import GradientLoss, FFTLoss, LaplacianLoss, BandMaskedFFTLoss
from src.stpc.data import PhysioNetDataset, ECGBeatDataset, EEGDataset, MaskedEEGDataset
from src.stpc.utils.ecg_utils import get_all_record_names, get_noise_signals, load_all_beats_from_dataset, load_and_resample_signal as load_ecg_signal
from src.stpc.utils.eeg_utils import load_eeg_from_edf, create_eeg_segments, get_adjacency_list, find_common_monopolar_channels

# --- Global Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
#                               ECG EXPERIMENTS
# ==============================================================================

def train_ecg_denoiser(config: dict):
    run_params = config["run_params"]
    train_params = config["training_params"]
    loss_weights = config["loss_weights"]
    paths = config["paths"]



    print("--- Training ECG Denoiser ---")
    print(f"Config: Gradient Loss={config['experiment_name']}, Gradient Loss={run_params['use_gradient_loss']}, FFT Loss={config["use_fft_loss"]}")
    os.makedirs(os.path.dirname(paths["save_path"]), exist_ok=True)

    class ECGConfig:
        LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS = 1e-4, 32, train_params["epochs"]
        SEGMENT_LENGTH_SAMPLES, SNR_DB_MIN, SNR_DB_MAX = 2048, -3, 12
        W_RECON, W_GRAD, W_FFT = 1.0, 0.5, 0.3

    config = ECGConfig()
    model = UNet1D(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=train_params['learning_rate'])
    
    print("Scanning for record names and loading noise signals...")
    all_record_names = get_all_record_names(paths['data_dir'])
    noise_signals = get_noise_signals(paths['noise_dir'], target_fs=250)

    loss_recon, loss_grad, loss_fft = nn.L1Loss(), GradientLoss(), FFTLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))

    for epoch in range(train_params['epochs']):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}: Loading a fresh subset of data...")
        records_for_epoch = np.random.choice(all_record_names, 20, replace=False)
        
        clean_signals = []
        for name in tqdm(records_for_epoch, desc="Loading signals"):
            signal = load_ecg_signal(os.path.join(paths['data_dir'], name), target_fs=250)
            if signal is not None and len(signal) > train_params['segment_samples']:
                clean_signals.append(signal)

        if not clean_signals:
            print("Warning: No usable signals loaded. Skipping epoch.")
            continue

        train_dataset = PhysioNetDataset(clean_signals, noise_signals, train_params, num_samples_per_epoch=5000)
        train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        
        loop = tqdm(train_loader, desc="Training")
        total_loss = 0.0
        for noisy, clean in loop:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE=='cuda')):
                denoised = model(noisy)
                loss = loss_recon(denoised, clean) * loss_weights['reconstruction']
                if run_params['use_gradient_loss']: 
                    loss += loss_weights['gradient'] * loss_grad(denoised, clean)
                if run_params['use_fft_loss']: 
                    loss += loss_weights['fft'] * loss_fft(denoised, clean)
            
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(loop):.6f}")
        torch.save(model.state_dict(), paths['save_path'])

def train_ecg_classifier(args):
    print("--- Training ECG Classifier ---")
    
    beats, labels = load_all_beats_from_dataset(args.data_dir)
    if len(beats) == 0: raise RuntimeError("Failed to extract beats.")

    X_train, X_val, y_train, y_val = train_test_split(beats, labels, test_size=0.2, random_state=42, stratify=labels)
    train_dataset = ECGBeatDataset(X_train, y_train)
    val_dataset = ECGBeatDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = ECGClassifier(num_classes=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for signals, targets in loop:
            signals, targets = signals.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, targets in val_loader:
                signals, targets = signals.to(DEVICE), targets.to(DEVICE)
                outputs = model(signals)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")
    
    torch.save(model.state_dict(), args.save_path)
    print(f"Classifier model saved to {args.save_path}")


# ==============================================================================
#                               EEG EXPERIMENTS
# ==============================================================================
def train_eeg_model(args):
    print(f"--- Starting EEG Training: {args.eeg_experiment_type} ---")
    
    # --- Config ---
    TARGET_FS, WINDOW_SECONDS = 256, 2
    WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_FS)
    SAMPLES_PER_EPOCH = 5000
    
    subject_dir = os.path.join(args.data_dir, 'chb01')
    common_channels = find_common_monopolar_channels(subject_dir)
    NUM_CHANNELS = len(common_channels)
    
    all_segments = []
    validation_file = 'chb01_03.edf'
    
    for f in tqdm(os.listdir(subject_dir), desc="Loading files"):
        if f.endswith('.edf'):
            if validation_file in f and args.eeg_experiment_type in ['spatial', 'frequency']:
                continue
            data, ch_names, _ = load_eeg_from_edf(os.path.join(subject_dir, f))
            if data is not None and set(common_channels).issubset(set(ch_names)):
                ch_map = {name: i for i, name in enumerate(ch_names)}
                ordered_data = data[[ch_map[name] for name in common_channels], :]
                all_segments.extend(create_eeg_segments(ordered_data, WINDOW_SAMPLES))
            
    if args.eeg_experiment_type == 'self_supervised':
        train_dataset = MaskedEEGDataset(all_segments, SAMPLES_PER_EPOCH)
    else:
        train_dataset = EEGDataset(all_segments, SAMPLES_PER_EPOCH)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np.savez(os.path.join(os.path.dirname(args.save_path), "norm_stats.npz"), 
             mean=train_dataset.mean, std=train_dataset.std, channels=common_channels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2, pin_memory=(DEVICE=='cuda'), shuffle=True)
    
    model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))
    
    loss_fns = {'L1': nn.L1Loss()}
    if args.eeg_experiment_type in ['spatial', 'self_supervised']:
        loss_fns['Spatial'] = LaplacianLoss(get_adjacency_list(common_channels))
    if args.eeg_experiment_type == 'spatial':
        loss_fns['Temporal'] = GradientLoss()
    elif args.eeg_experiment_type == 'frequency':
        loss_fns.update({
            'Alpha': BandMaskedFFTLoss(TARGET_FS, 8, 12),
            'Low': BandMaskedFFTLoss(TARGET_FS, 0.5, 8),
            'High': BandMaskedFFTLoss(TARGET_FS, 12, 70)
        })

    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0.0
        for data_in, data_clean in loop:
            data_in, data_clean = data_in.to(DEVICE), data_clean.to(DEVICE)
            with torch.autocast(device_type=DEVICE, enabled=(DEVICE=='cuda')):
                reconstructed = model(data_in)
                loss = loss_fns['L1'](reconstructed, data_clean)
                if args.eeg_experiment_type == 'spatial':
                    loss += args.alpha * loss_fns['Temporal'](reconstructed, data_clean) + \
                            args.beta * loss_fns['Spatial'](reconstructed, data_clean)
                elif args.eeg_experiment_type == 'frequency':
                    loss += 5.0 * loss_fns['Alpha'](reconstructed, data_clean) + \
                            0.1 * loss_fns['Low'](reconstructed, data_clean) + \
                            0.1 * loss_fns['High'](reconstructed, data_clean)
                elif args.eeg_experiment_type == 'self_supervised':
                    loss += args.beta * loss_fns['Spatial'](reconstructed, data_clean)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(loop):.6f}")
        torch.save(model.state_dict(), args.save_path)
    print(f"✅ Training complete for {args.eeg_experiment_type}. Model saved to {args.save_path}")


# ==============================================================================
#                               MAIN SCRIPT LOGIC
# ==============================================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Unified Training Runner for STPC Experiments")
#     subparsers = parser.add_subparsers(dest="experiment", required=True)

#     p_ecg_denoise = subparsers.add_parser("ecg_denoiser", help="Train ECG denoiser models.")
#     p_ecg_denoise.add_argument("--data_dir", type=str, required=True)
#     p_ecg_denoise.add_argument("--noise_dir", type=str, required=True)
#     p_ecg_denoise.add_argument("--save_path", type=str, required=True)
#     p_ecg_denoise.add_argument("--epochs", type=int, default=10)
#     p_ecg_denoise.add_argument('--no-gradient-loss', dest='use_gradient_loss', action='store_false')
#     p_ecg_denoise.add_argument('--no-fft-loss', dest='use_fft_loss', action='store_false')
#     p_ecg_denoise.set_defaults(use_gradient_loss=True, use_fft_loss=True)

#     p_ecg_class = subparsers.add_parser("ecg_classifier", help="Train the ECG beat classifier.")
#     p_ecg_class.add_argument("--data_dir", type=str, required=True)
#     p_ecg_class.add_argument("--save_path", type=str, required=True)
#     p_ecg_class.add_argument("--epochs", type=int, default=5)

#     p_eeg = subparsers.add_parser("eeg", help="Train EEG models.")
#     p_eeg.add_argument("--eeg_experiment_type", type=str, required=True,
#                        choices=['baseline', 'spatial', 'frequency', 'self_supervised'])
#     p_eeg.add_argument("--data_dir", type=str, required=True)
#     p_eeg.add_argument("--save_path", type=str, required=True)
#     p_eeg.add_argument("--epochs", type=int, default=10)
#     p_eeg.add_argument('--alpha', type=float, default=1.0)
#     p_eeg.add_argument('--beta', type=float, default=1.0)
    
#     args = parser.parse_args()

#     if args.experiment == "ecg_denoiser":
#         train_ecg_denoiser(args)
#     elif args.experiment == "ecg_classifier":
#         train_ecg_classifier(args)
#     elif args.experiment == "eeg":
#         train_eeg_model(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Training Runner for STPC Experiments")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=""
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    
    experiment = config.get('experiment_name', '')

    if "ecg_denoiser" in experiment:
        train_ecg_denoiser(config)
    elif "ecg_classifier" in experiment:
        print()
    elif "eeg" in experiment:
        print()
    else:
        print("Unknown expirment name")