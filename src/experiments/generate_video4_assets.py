# src/experiments/generate_video4_assets.py

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Add project root to system path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import from our STPC library ---
from src.stpc.model import UNet1D
from src.stpc.utils.eeg_utils import load_eeg_from_edf, TARGET_FS as EEG_FS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def generate_assets(args):
    """
    Generates all necessary data arrays (.npy) and plots (.png)
    for Video 4: "The Denoising Delusion".
    """
    print("--- Starting Asset Generation for Video 4 ---")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. Load Models ---
    print("Loading trained models...")
    stats_path = os.path.join(os.path.dirname(args.l1_model_path), "norm_stats.npz")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Normalization stats file not found at {stats_path}")
    
    stats = np.load(stats_path)
    mean, std = stats['mean'], stats['std']
    NUM_CHANNELS = len(stats['channels'])

    l1_model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(DEVICE)
    l1_model.load_state_dict(torch.load(args.l1_model_path, map_location=DEVICE))
    l1_model.eval()

    stpc_model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(DEVICE)
    stpc_model.load_state_dict(torch.load(args.stpc_model_path, map_location=DEVICE))
    stpc_model.eval()
    print("Models loaded successfully.")

    # --- 2. Load and Prepare Data ---
    print("Loading and preparing EEG seizure data...")
    eeg_file_path = os.path.join(args.data_dir, "chb01/chb01_03.edf")
    clean_data, final_ch_names, _ = load_eeg_from_edf(eeg_file_path)
    
    # Isolate the sharp seizure spike (same segment as the paper)
    seizure_start_s = 2996
    segment_duration_s = 4
    spike_start_sample = int(seizure_start_s * EEG_FS)
    spike_end_sample = spike_start_sample + int(segment_duration_s * EEG_FS)
    
    ground_truth_segment = clean_data[:, spike_start_sample:spike_end_sample]

    # Create a reproducible, highly noisy version
    np.random.seed(42)
    noise = np.random.randn(*ground_truth_segment.shape).astype(np.float32) * np.std(ground_truth_segment) * 2.0
    noisy_segment = ground_truth_segment + noise
    print("Data prepared.")

    # --- 3. Run Inference ---
    print("Running inference with both models...")
    with torch.no_grad():
        noisy_norm = (noisy_segment - mean) / (std + 1e-8)
        noisy_tensor = torch.from_numpy(noisy_norm).float().unsqueeze(0).to(DEVICE)
        
        # Denoise with L1 model
        l1_output_norm = l1_model(noisy_tensor).squeeze(0).cpu().numpy()
        l1_denoised_segment = (l1_output_norm * (std + 1e-8)) + mean
        
        # Denoise with STPC model
        stpc_output_norm = stpc_model(noisy_tensor).squeeze(0).cpu().numpy()
        stpc_denoised_segment = (stpc_output_norm * (std + 1e-8)) + mean
    print("Inference complete.")

    # --- 4. Save Data for Manim ---
    # We will use a representative channel for the animation, e.g., channel 8 (C4-P4)
    channel_idx = 8
    print(f"Saving .npy data arrays for Manim (from channel {final_ch_names[channel_idx]})...")
    np.save(os.path.join(args.output_dir, 'gt_wave.npy'), ground_truth_segment[channel_idx])
    np.save(os.path.join(args.output_dir, 'noisy_wave.npy'), noisy_segment[channel_idx])
    np.save(os.path.join(args.output_dir, 'l1_denoised_wave.npy'), l1_denoised_segment[channel_idx])
    np.save(os.path.join(args.output_dir, 'stpc_denoised_wave.npy'), stpc_denoised_segment[channel_idx])
    
    # Save gradients as well
    np.save(os.path.join(args.output_dir, 'gt_grad.npy'), np.diff(ground_truth_segment[channel_idx]))
    np.save(os.path.join(args.output_dir, 'l1_denoised_grad.npy'), np.diff(l1_denoised_segment[channel_idx]))
    np.save(os.path.join(args.output_dir, 'stpc_denoised_grad.npy'), np.diff(stpc_denoised_segment[channel_idx]))
    print("Data arrays saved.")

    # --- 5. Generate High-Quality Plots for Video ---
    print("Generating high-quality plots...")
    time_axis = np.arange(ground_truth_segment.shape[1]) / EEG_FS
    
    # Plot 1: Waveform Comparison
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(time_axis, noisy_segment[channel_idx], color='gray', alpha=0.5, label='Noisy Input')
    ax.plot(time_axis, l1_denoised_segment[channel_idx], 'r--', linewidth=2, label='Denoised (L1 Only)')
    ax.plot(time_axis, stpc_denoised_segment[channel_idx], color='lime', linewidth=2.5, label='Denoised (Full STPC)')
    ax.plot(time_axis, ground_truth_segment[channel_idx], 'white', linewidth=2.5, label='Ground Truth')
    ax.set_title("Waveform Comparison: Seizure Spike Denoising", fontsize=20, pad=20)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Amplitude (uV)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'video4_waveform_comparison.png'), dpi=300)
    
    # Plot 2: Gradient Comparison
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(time_axis[1:], np.diff(l1_denoised_segment[channel_idx]), 'r--', linewidth=2, label='L1 Gradient (Incorrect)')
    ax.plot(time_axis[1:], np.diff(stpc_denoised_segment[channel_idx]), color='lime', linewidth=2.5, label='STPC Gradient (Correct)')
    ax.plot(time_axis[1:], np.diff(ground_truth_segment[channel_idx]), 'white', linewidth=2.5, label='Ground Truth Gradient')
    ax.set_title("Gradient Comparison: Capturing Signal Dynamics", fontsize=20, pad=20)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Gradient (uV/sample)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'video4_gradient_comparison.png'), dpi=300)
    
    plt.style.use('default') # Reset style
    print("Plots generated successfully.")
    print(f"✅ All assets saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate assets for Video 4 of the 'Signal & State' series.")
    parser.add_argument("--l1_model_path", type=str, required=True, help="Path to the trained L1-only EEG model (.pth).")
    parser.add_argument("--stpc_model_path", type=str, required=True, help="Path to the trained Full STPC EEG model (.pth).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the root of the CHB-MIT EEG database.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated .npy and .png files.")
    
    args = parser.parse_args()
    generate_assets(args)