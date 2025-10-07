# src/experiments/generate_academic_comparison.py

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# --- Add project root to system path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import from our STPC library ---
from src.stpc.model import UNet1D
from src.stpc.utils.eeg_utils import load_eeg_from_edf, create_eeg_segments, TARGET_FS as EEG_FS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Helper function for plotting
def setup_dark_plot(figsize=(15, 7), title="", xlabel="", ylabel=""):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=20, pad=20)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, alpha=0.2)
    return fig, ax

def main(args):
    print("--- Starting Academic Comparison Asset Generation ---")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load Models and Normalization Stats ---
    print("Loading models and normalization stats...")
    stats_path = os.path.join(os.path.dirname(args.stpc_model_path), "norm_stats.npz")
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

    # ===================================================================
    #   ANALYSIS 1: RECONSTRUCTION FIDELITY
    # ===================================================================
    print("\n--- Analysis 1: Generating Reconstruction Fidelity Plots ---")
    eeg_file_path = os.path.join(args.data_dir, "chb01/chb01_03.edf")
    clean_data, final_ch_names, _ = load_eeg_from_edf(eeg_file_path)
    
    spike_start_sample = int(2996 * EEG_FS)
    spike_end_sample = spike_start_sample + int(4 * EEG_FS)
    ground_truth_segment = clean_data[:, spike_start_sample:spike_end_sample]

    np.random.seed(42)
    noise = np.random.randn(*ground_truth_segment.shape).astype(np.float32) * np.std(ground_truth_segment) * 2.0
    noisy_segment = ground_truth_segment + noise
    
    with torch.no_grad():
        noisy_norm = (noisy_segment - mean) / (std + 1e-8)
        noisy_tensor = torch.from_numpy(noisy_norm).float().unsqueeze(0).to(DEVICE)
        l1_denoised_segment = ((l1_model(noisy_tensor).squeeze(0).cpu().numpy() * std) + mean)
        stpc_denoised_segment = ((stpc_model(noisy_tensor).squeeze(0).cpu().numpy() * std) + mean)

    time_axis = np.arange(ground_truth_segment.shape[1]) / EEG_FS
    channel_idx = 8 # Representative channel
    
    # Plot 1: Waveform
    fig, ax = setup_dark_plot(title="Reconstruction Fidelity: Waveform", xlabel="Time (s)", ylabel="Amplitude (uV)")
    ax.plot(time_axis, noisy_segment[channel_idx], color='gray', alpha=0.5, label='Noisy Input')
    ax.plot(time_axis, l1_denoised_segment[channel_idx], 'r--', linewidth=2, label='Denoised (L1 Only)')
    ax.plot(time_axis, stpc_denoised_segment[channel_idx], color='lime', linewidth=2.5, label='Denoised (STPC)')
    ax.plot(time_axis, ground_truth_segment[channel_idx], 'white', linewidth=2.5, label='Ground Truth')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, '1_recon_waveform.png'), dpi=300)
    
    # Plot 2: Gradient
    fig, ax = setup_dark_plot(title="Reconstruction Fidelity: Signal Dynamics (Gradient)", xlabel="Time (s)", ylabel="Gradient (uV/sample)")
    ax.plot(time_axis[1:], np.diff(l1_denoised_segment[channel_idx]), 'r--', linewidth=2, label='L1 Gradient (Chaotic)')
    ax.plot(time_axis[1:], np.diff(stpc_denoised_segment[channel_idx]), color='lime', linewidth=2.5, label='STPC Gradient (Plausible)')
    ax.plot(time_axis[1:], np.diff(ground_truth_segment[channel_idx]), 'white', linewidth=2.5, label='Ground Truth Gradient')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, '2_recon_gradient.png'), dpi=300)
    print("Reconstruction plots saved.")

    # ===================================================================
    #   ANALYSIS 2: REPRESENTATION QUALITY (THE "WOW" FACTOR)
    # ===================================================================
    print("\n--- Analysis 2: Generating Representation Quality Plots (UMAP) ---")
    seizure_data, _, _ = load_eeg_from_edf(os.path.join(args.data_dir, 'chb01/chb01_03.edf'))
    non_seizure_data, _, _ = load_eeg_from_edf(os.path.join(args.data_dir, 'chb01/chb01_01.edf'))

    seizure_segs = create_eeg_segments(seizure_data, 2 * EEG_FS)
    non_seizure_segs = create_eeg_segments(non_seizure_data, 2 * EEG_FS)
    
    test_segments = np.array(non_seizure_segs[850:1050] + seizure_segs[1400:1600])
    test_labels = np.array([0]*200 + [1]*200) # 0=Non-Seizure, 1=Seizure
    
    test_segments_norm = (test_segments - mean) / (std + 1e-8)
    
    # Extract embeddings for both models
    all_embeddings = {'l1': [], 'stpc': []}
    for i in tqdm(range(len(test_segments_norm)), desc="Extracting embeddings"):
        segment_tensor = torch.from_numpy(test_segments_norm[i]).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            all_embeddings['l1'].append(l1_model.encode(segment_tensor).cpu().numpy())
            all_embeddings['stpc'].append(stpc_model.encode(segment_tensor).cpu().numpy())
            
    all_embeddings['l1'] = np.concatenate(all_embeddings['l1'])
    all_embeddings['stpc'] = np.concatenate(all_embeddings['stpc'])
    
    # Generate UMAP plots for both
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    
    for model_name in ['l1', 'stpc']:
        print(f"Generating UMAP for {model_name.upper()} model...")
        embedding_2d = reducer.fit_transform(all_embeddings[model_name])
        score = silhouette_score(embedding_2d, test_labels)
        
        fig, ax = setup_dark_plot(figsize=(10, 8), title=f"Latent Space: {model_name.upper()} Model\nSilhouette Score: {score:.4f}")
        scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=test_labels, cmap='viridis', s=15, alpha=0.7)
        ax.legend(handles=scatter.legend_elements()[0], labels=['Non-Seizure', 'Seizure'], fontsize=12)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'3_umap_{model_name}.png'), dpi=300)
    print("UMAP plots saved.")
    
    plt.style.use('default') # Reset style
    print("\n✅ All assets generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a visual abstract comparing L1 and STPC models.")
    parser.add_argument("--l1_model_path", type=str, required=True, help="Path to the trained L1-only EEG model (.pth).")
    parser.add_argument("--stpc_model_path", type=str, required=True, help="Path to the trained Full STPC EEG model (.pth).")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the root of the CHB-MIT EEG database.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated .png files.")
    
    args = parser.parse_args()
    main(args)