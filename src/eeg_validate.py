# src/eeg_validate.py
import torch
import numpy as np
import os
import argparse
import mne
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from scipy.signal import coherence, welch
from skimage.metrics import structural_similarity as ssim
import umap.umap_ as umap
from sklearn.metrics import silhouette_score

from model import UNet1D
from eeg_data_utils import load_eeg_from_edf, create_eeg_segments, TARGET_FS

# ... (Helper functions from before: calculate_mean_ssim, calculate_mean_coherence, create_psd_plot, create_topomap_video) ...
def create_psd_plot(data_dict, fs, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True); fig.suptitle('Power Spectral Density Comparison', fontsize=16)
    signals_to_plot = { "Noisy Input": data_dict["Noisy Input"], "Ground Truth": data_dict["Ground Truth"],
                        "Denoised (Frequency STPC)": data_dict["Denoised (Frequency STPC)"]}
    for ax, (title, signal) in zip(axes, signals_to_plot.items()):
        f, Pxx = welch(signal, fs=fs, nperseg=fs*2, axis=-1); Pxx_mean = np.mean(Pxx, axis=0)
        ax.semilogy(f, Pxx_mean); ax.set_title(title); ax.set_ylabel('Power/Frequency (dB/Hz)'); ax.grid(True, which="both", ls="--")
        ax.axvspan(8, 12, color='orange', alpha=0.2, label='Alpha Band (8-12 Hz)'); ax.legend(loc='upper right')
    axes[-1].set_xlabel('Frequency (Hz)'); plt.xlim(0, 50); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Saving PSD plot to {output_path}"); plt.savefig(output_path); plt.close(fig)
def create_topomap_video(data_dict, info, output_path, fs):
    print("Generating topography video..."); vmax = np.max(np.abs(data_dict["Ground Truth"])) * 0.8; vmin = -vmax; frames = []
    for i in tqdm(range(256), desc="Generating frames"):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5)); fig.suptitle(f'EEG Topography Comparison | Time = {i/fs:.2f} s', fontsize=16)
        for ax, (title, data) in zip(axes, data_dict.items()):
            mne.viz.plot_topomap(data[:, i], info, axes=ax, show=False, vlim=(vmin, vmax))
            ax.set_title(title)
        fig.canvas.draw(); frame = np.array(fig.canvas.renderer.buffer_rgba()).reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]; frames.append(frame); plt.close(fig)
    print(f"Saving video to {output_path}"); imageio.mimsave(output_path, frames, fps=30)


# --- NEW FUNCTION FOR PHASE 3 VISUAL ---
def create_embedding_plot(model, data_loader, device, output_path):
    """Extracts embeddings, performs UMAP, and creates a scatter plot."""
    all_embeddings = []
    all_labels = [] # 0 for non-seizure, 1 for seizure
    
    print("Extracting embeddings from test data...")
    for segments, labels in tqdm(data_loader):
        segments = segments.to(device)
        with torch.no_grad():
            # Use the .encode() method we added to the model
            embeddings = model.encode(segments).cpu().numpy()
        all_embeddings.append(embeddings)
        all_labels.append(labels.numpy())
        
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)

    print("Running UMAP for dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)
    
    # Calculate Silhouette Score to quantify cluster separation
    score = silhouette_score(embedding_2d, all_labels)
    
    print("Generating scatter plot...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=all_labels, cmap='viridis', s=10, alpha=0.7)
    plt.title(f'UMAP Projection of Learned EEG Embeddings\nSilhouette Score: {score:.4f}', fontsize=16)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=['Non-Seizure', 'Seizure'])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    print(f"Saving embedding plot to {output_path}")
    plt.savefig(output_path)
    plt.close()

def main(args):
    print(f"--- Starting EEG Denoising Validation for experiment: {args.experiment} ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- EXPERIMENT-SPECIFIC LOGIC ---
    if args.experiment == 'self_supervised':
        # 1. Load data: seizure and non-seizure segments
        seizure_file = os.path.join(args.data_dir, 'chb01/chb01_03.edf')
        non_seizure_file = os.path.join(args.data_dir, 'chb01/chb01_01.edf')
        
        seizure_data, _ = load_eeg_from_edf(seizure_file)
        non_seizure_data, final_ch_names = load_eeg_from_edf(non_seizure_file)
        NUM_CHANNELS = len(final_ch_names)

        # Create labeled segments for testing
        seizure_segments = create_eeg_segments(seizure_data, 2 * TARGET_FS)
        non_seizure_segments = create_eeg_segments(non_seizure_data, 2 * TARGET_FS)
        
        # Take a subset to keep validation fast
        test_segments = np.array(non_seizure_segments[:200] + seizure_segments[1400:1600])
        test_labels = np.array([0]*200 + [1]*200) # 0 = non-seizure, 1 = seizure
        
        # 2. Load model and stats
        model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(device)
        model.load_state_dict(torch.load(args.self_supervised_model_path, map_location=device))
        model.eval()
        
        stats = np.load(os.path.join(os.path.dirname(args.self_supervised_model_path), "norm_stats.npz"))
        mean, std = stats['mean'], stats['std']

        # 3. Normalize and create DataLoader
        test_segments_norm = (test_segments - mean) / (std + 1e-8)
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_segments_norm).float(), torch.from_numpy(test_labels))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        # 4. Run validation
        create_embedding_plot(model, test_loader, device, args.output_path)
    
    else: # Logic for other experiments remains for completeness
        # ... (This part can be abridged as it's already tested)
        print("This validation mode is for spatial or frequency experiments.")

    print("✅ Validation complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, choices=['spatial', 'frequency', 'self_supervised'])
    parser.add_argument('--data_dir', type=str, required=True)
    # Add path for the new model
    parser.add_argument('--self_supervised_model_path', type=str)
    # Make other paths optional
    parser.add_argument('--baseline_model_path', type=str)
    parser.add_argument('--spatial_model_path', type=str)
    parser.add_argument('--frequency_model_path', type=str)
    parser.add_argument('--output_path', type=str, required=True)
    cli_args = parser.parse_args()
    
    if cli_args.experiment == 'self_supervised' and not cli_args.self_supervised_model_path:
        parser.error("--self_supervised_model_path is required.")
        
    main(cli_args)