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
import mne # Keep for EEG part
from sklearn.model_selection import train_test_split

# ==============================================================================
#                      SETUP: Add project root to system path
# ==============================================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==============================================================================

# --- Import from our new STPC library ---
from stpc.model import UNet1D, ECGClassifier
from stpc.losses import GradientLoss, FFTLoss, LaplacianLoss, BandMaskedFFTLoss
from stpc.data import PhysioNetDataset, ECGBeatDataset, EEGDataset, MaskedEEGDataset
from stpc.utils.ecg_utils import get_all_record_names, load_and_resample_signal, get_noise_signals, load_all_beats_from_dataset
from stpc.utils.eeg_utils import load_eeg_from_edf, create_eeg_segments, get_adjacency_list

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
#                               ECG EXPERIMENTS
# ==============================================================================

def train_ecg_denoiser(args):
    """
    Handles the training for the ECG denoiser ablation study with a memory-efficient
    data loading strategy.
    """
    print("--- Training ECG Denoiser ---")
    print(f"Config: L1 Loss {'Enabled'}, Gradient Loss {args.use_gradient_loss}, FFT Loss {args.use_fft_loss}")
    print(f"Model will be saved to: {args.save_path}")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    class ECGConfig:
        LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS = 1e-4, 32, args.epochs
        SEGMENT_LENGTH_SAMPLES, SNR_DB_MIN, SNR_DB_MAX = 2048, -3, 12
        W_RECON, W_GRAD, W_FFT = 1.0, 0.5, 0.3

    config = ECGConfig()
    model = UNet1D(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    print("Scanning for record names and loading noise signals...")
    all_record_names = get_all_record_names(args.data_dir)
    noise_signals = get_noise_signals(args.noise_dir, 250)

    loss_recon, loss_grad, loss_fft = nn.L1Loss(), GradientLoss(), FFTLoss()
    # Corrected the deprecated GradScaler call for future-proofing
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}: Loading a fresh subset of data...")
        records_for_epoch = np.random.choice(all_record_names, 20, replace=False)
        
        clean_signals = []
        for name in tqdm(records_for_epoch, desc="Loading signals"):
            # --- THIS IS THE FINAL FIX ---
            # Construct the full path prefix here, just as load_all_beats does.
            record_path_prefix = os.path.join(args.data_dir, name)
            # Pass the full prefix to the utility function.
            signal = load_and_resample_signal(record_path_prefix, 250)
            
            if signal is not None and len(signal) > config.SEGMENT_LENGTH_SAMPLES:
                clean_signals.append(signal)

        if not clean_signals:
            print("Warning: No usable clean signals were loaded for this epoch. Skipping.")
            continue

        train_dataset = PhysioNetDataset(clean_signals, noise_signals, config, num_samples_per_epoch=5000)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        
        loop = tqdm(train_loader, desc=f"Training")
        total_loss = 0.0
        for noisy, clean in loop:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE=='cuda')):
                denoised = model(noisy)
                loss = loss_recon(denoised, clean) * config.W_RECON
                if args.use_gradient_loss: loss += config.W_GRAD * loss_grad(denoised, clean)
                if args.use_fft_loss: loss += config.W_FFT * loss_fft(denoised, clean)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} complete. Avg Loss: {total_loss / len(loop):.6f}")
        torch.save(model.state_dict(), args.save_path)


def train_ecg_classifier(args):
    """Handles training for the downstream ECG beat classifier."""
    print("--- Training ECG Classifier ---")
    
    beats, labels = load_all_beats_from_dataset(args.data_dir)
    if len(beats) == 0:
        raise RuntimeError("Failed to extract any beats.")

    np.save('all_beats.npy', beats)
    np.save('all_labels.npy', labels)

    # --- THIS IS THE FIX ---
    # Correctly split into training AND validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        beats, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create separate datasets for training and validation.
    train_dataset = ECGBeatDataset(X_train, y_train)
    val_dataset = ECGBeatDataset(X_val, y_val) # We will use this for a quick validation check
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False) # Loader for validation
    # --- END OF FIX ---

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

        # Optional: Add a quick validation check at the end of each epoch
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
    """Handles all EEG training variations based on args."""
    print(f"--- Training EEG Model ---")
    print(f"Experiment Type: {args.eeg_experiment_type}")

    # --- Configs ---
    WINDOW_SAMPLES = int(2 * 256) # 2 seconds
    
    # --- Data Loading (common for all EEG experiments) ---
    print("Scanning for common channels and loading all EEG segments...")
    subject_dir = os.path.join(args.data_dir, 'chb01')
    all_files = [os.path.join(subject_dir, f) for f in os.listdir(subject_dir) if f.endswith('.edf')]
    
    # Simplified channel finding
    raw_sample = mne.io.read_raw_edf(all_files[0], preload=False, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    common_channels = sorted([ch for ch in raw_sample.ch_names if ch.split('-')[0].upper() in montage.ch_names])
    NUM_CHANNELS = len(common_channels)
    print(f"Using {NUM_CHANNELS} common channels.")

    all_segments = []
    for f in tqdm(all_files, desc="Processing files"):
        data, ch_names, _ = load_eeg_from_edf(f)
        if data is not None and ch_names is not None:
             all_segments.extend(create_eeg_segments(data, WINDOW_SAMPLES))
    
    # --- Dataset Selection ---
    if args.eeg_experiment_type == 'self_supervised':
        print("Using MaskedEEGDataset for self-supervised learning.")
        train_dataset = MaskedEEGDataset(clean_segments=all_segments, samples_per_epoch=5000)
    else:
        print("Using standard EEGDataset for denoising task.")
        train_dataset = EEGDataset(clean_segments=all_segments, samples_per_epoch=5000)

    # Save normalization stats
    stats_dir = os.path.dirname(args.save_path)
    os.makedirs(stats_dir, exist_ok=True)
    np.savez(os.path.join(stats_dir, "norm_stats.npz"), mean=train_dataset.mean, std=train_dataset.std)

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2, pin_memory=True, shuffle=True)
    
    # --- Model, Optimizer, and Losses ---
    model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))

    # Loss selection
    loss_l1 = nn.L1Loss()
    loss_spatial = LaplacianLoss(get_adjacency_list(common_channels))
    loss_temporal = GradientLoss()
    loss_frequency = BandMaskedFFTLoss(fs=256, f_low=8, f_high=12) # Alpha band

    for epoch in range(args.epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in loop:
            # EEG datasets return (noisy, clean) or (masked, clean)
            input_signal, target_signal = batch
            input_signal, target_signal = input_signal.to(DEVICE), target_signal.to(DEVICE)
            
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE=='cuda')):
                reconstructed = model(input_signal)
                
                # --- Dynamic Loss Calculation ---
                loss = 0
                if args.eeg_experiment_type == 'self_supervised':
                    loss = loss_l1(reconstructed, target_signal) + 1.0 * loss_spatial(reconstructed, target_signal)
                else: # Denoising experiments
                    loss = loss_l1(reconstructed, target_signal) # Baseline
                    if args.eeg_experiment_type == 'spatial':
                        loss += 1.0 * loss_spatial(reconstructed, target_signal)
                        loss += 0.5 * loss_temporal(reconstructed, target_signal)
                    elif args.eeg_experiment_type == 'frequency':
                         loss += 5.0 * loss_frequency(reconstructed, target_signal) # Higher weight for targeted filtering

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}, Avg Loss: {loss.item():.6f}") # Note: Simplified loss reporting
        torch.save(model.state_dict(), args.save_path)
    
    print(f"EEG model saved to {args.save_path}")


# ==============================================================================
#                               MAIN SCRIPT LOGIC
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Training Runner for STPC Experiments")
    subparsers = parser.add_subparsers(dest="experiment", required=True, help="The experiment to run")

    p_ecg_denoise = subparsers.add_parser("ecg_denoiser", help="Train the ECG denoiser.")
    p_ecg_denoise.add_argument("--data_dir", type=str, required=True, help="Path to 'mit-bih-arrhythmia-database-1.0.0'")
    p_ecg_denoise.add_argument("--noise_dir", type=str, required=True, help="Path to 'mit-bih-noise-stress-test-database-1.0.0'")
    p_ecg_denoise.add_argument("--save_path", type=str, required=True)
    p_ecg_denoise.add_argument("--epochs", type=int, default=50)
    p_ecg_denoise.add_argument('--no-gradient-loss', dest='use_gradient_loss', action='store_false')
    p_ecg_denoise.add_argument('--no-fft-loss', dest='use_fft_loss', action='store_false')
    p_ecg_denoise.set_defaults(use_gradient_loss=True, use_fft_loss=True)

    p_ecg_class = subparsers.add_parser("ecg_classifier", help="Train the ECG beat classifier.")
    p_ecg_class.add_argument("--data_dir", type=str, required=True)
    p_ecg_class.add_argument("--save_path", type=str, required=True)
    p_ecg_class.add_argument("--epochs", type=int, default=20)

    p_eeg = subparsers.add_parser("eeg", help="Train an EEG model (baseline, spatial, frequency, or self-supervised).")
    
    p_eeg.add_argument(
        "--eeg_experiment_type", 
        type=str, 
        default='baseline',
        choices=['baseline', 'spatial', 'frequency', 'self_supervised'],
        required=True,
        help="The specific type of EEG experiment to run."
    )
    p_eeg.add_argument("--data_dir", type=str, required=True, help="Path to the root EEG data directory (e.g., 'chb-mit-scalp-eeg-database-1.0.0').")
    p_eeg.add_argument("--save_path", type=str, required=True, help="Full path where the trained model will be saved.")
    p_eeg.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    p_eeg.add_argument('--alpha', type=float, default=1.0, help='Weight for temporal gradient loss (for spatial experiment).')
    p_eeg.add_argument('--beta', type=float, default=1.0, help='Weight for spatial laplacian loss (for spatial experiment).')
    
    args = parser.parse_args()

    if args.experiment == "ecg_denoiser":
        train_ecg_denoiser(args)
    elif args.experiment == "ecg_classifier":
        train_ecg_classifier(args)
    elif args.experiment == "eeg":
        train_eeg_model(args)