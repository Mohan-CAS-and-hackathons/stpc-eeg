# src/eeg_validate.py
import torch
import numpy as np
import os
import argparse
import mne
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from scipy.signal import coherence
from skimage.metrics import structural_similarity as ssim

from model import UNet1D
from eeg_data_utils import load_eeg_from_edf, TARGET_FS, get_adjacency_list

# --- NEW HELPER FUNCTIONS FOR ADVANCED METRICS ---

def calculate_mean_ssim(clean_topo_series, denoised_topo_series, info):
    """Calculates the average Structural Similarity Index (SSIM) between two series of topomaps."""
    ssim_scores = []
    
    # --- DEFINITIVE FIX: Create a larger, higher-DPI figure for rendering ---
    # This ensures the rendered numpy arrays are larger than the SSIM window size.
    fig = plt.figure(figsize=(2, 2), dpi=100) # figsize in inches, dpi=dots per inch
    # --- END FIX ---
    
    data_range = np.max(clean_topo_series) - np.min(clean_topo_series)

    # We only need to check a subset of frames for a reliable estimate
    num_frames_to_check = 64
    time_indices = np.linspace(0, clean_topo_series.shape[1] - 1, num_frames_to_check).astype(int)

    for i in tqdm(time_indices, desc="Calculating SSIM frames"):
        ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
        
        mne.viz.plot_topomap(clean_topo_series[:, i], info, axes=ax1, show=False)
        fig.canvas.draw()
        clean_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        
        mne.viz.plot_topomap(denoised_topo_series[:, i], info, axes=ax2, show=False)
        fig.canvas.draw()
        denoised_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        
        # multichannel=True was deprecated, use channel_axis=-1 instead for new scikit-image versions
        score = ssim(clean_img, denoised_img, channel_axis=-1, data_range=255)
        ssim_scores.append(score)
        
        fig.clear()
        
    plt.close(fig)
    return np.mean(ssim_scores)


def calculate_mean_coherence(clean_signal, denoised_signal, fs=TARGET_FS):
    """Calculates the average magnitude-squared coherence across standard EEG bands."""
    bands = {
        'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12),
        'Beta': (12, 30), 'Gamma': (30, 70)
    }
    avg_coherence = 0
    
    for band_name, (fmin, fmax) in bands.items():
        # Calculate coherence for each channel
        band_coherence_scores = []
        for ch in range(clean_signal.shape[0]):
            f, Cxy = coherence(clean_signal[ch, :], denoised_signal[ch, :], fs=fs, nperseg=fs) # 1-second windows
            # Find frequency indices within the band
            idx_band = np.where((f >= fmin) & (f <= fmax))
            # Average coherence within the band for this channel
            if len(idx_band[0]) > 0:
                mean_c_ch = np.mean(Cxy[idx_band])
                band_coherence_scores.append(mean_c_ch)
        
        # Average across channels for this band
        if band_coherence_scores:
            avg_coherence += np.mean(band_coherence_scores)
            
    return avg_coherence / len(bands)


def validate_and_visualize(args):
    """
    Loads models, denoises a segment, and calculates a full suite of
    quantitative metrics (RMSE, SSIM, Coherence) before generating visuals.
    """
    print("--- Starting EEG Denoising Validation ---")
    
    clean_data, final_ch_names = load_eeg_from_edf(args.test_file)
    if clean_data is None: print("Failed to load data."); return
        
    NUM_CHANNELS = clean_data.shape[0]
    print(f"Data loaded with {NUM_CHANNELS} unique monopolar channels.")

    seizure_start_sample = 2996 * TARGET_FS
    segment_length = 4 * TARGET_FS
    clean_segment = clean_data[:, seizure_start_sample : seizure_start_sample + segment_length]
    noise = np.random.randn(*clean_segment.shape).astype(np.float32) * np.std(clean_segment) * 1.5
    noisy_segment = clean_segment + noise
    
    stats_path = os.path.join(os.path.dirname(args.baseline_model_path), "norm_stats.npz")
    if not os.path.exists(stats_path): print(f"FATAL: Stats file not found at {stats_path}."); return
    stats = np.load(stats_path)
    mean, std, saved_channels = stats['mean'], stats['std'], stats['channels']
    if not np.array_equal(saved_channels, final_ch_names): print("FATAL: Channel mismatch."); return
    print("Normalization stats loaded.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    def load_model(path, in_ch, out_ch, dev):
        model = UNet1D(in_channels=in_ch, out_channels=out_ch).to(dev)
        if path and os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=dev))
            print(f"Model loaded from {path}.")
        else: return None
        model.eval()
        return model

    baseline_model = load_model(args.baseline_model_path, NUM_CHANNELS, NUM_CHANNELS, device)
    spatial_model = load_model(args.spatial_model_path, NUM_CHANNELS, NUM_CHANNELS, device)
    
    noisy_segment_norm = (noisy_segment - mean) / (std + 1e-8)
    noisy_tensor = torch.from_numpy(noisy_segment_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_baseline_norm = baseline_model(noisy_tensor).squeeze(0).cpu().numpy() if baseline_model else np.zeros_like(noisy_segment_norm)
        denoised_spatial_norm = spatial_model(noisy_tensor).squeeze(0).cpu().numpy() if spatial_model else np.zeros_like(noisy_segment_norm)
    print("Denoising complete.")
    
    denoised_baseline = (denoised_baseline_norm * (std + 1e-8)) + mean
    denoised_spatial = (denoised_spatial_norm * (std + 1e-8)) + mean

    # --- Quantitative Analysis ---
    print("\n" + "="*50 + "\n      Quantitative Performance Metrics\n" + "="*50)
    
    # Info object needed for SSIM
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=final_ch_names, sfreq=TARGET_FS, ch_types='eeg')
    info.set_montage(montage, on_missing='ignore')

    # Metric 1: RMSE
    rmse_baseline = np.sqrt(np.mean((denoised_baseline - clean_segment)**2))
    rmse_spatial = np.sqrt(np.mean((denoised_spatial - clean_segment)**2))
    
    # Metric 2: SSIM
    print("Calculating Mean SSIM (this will take a moment)...")
    ssim_baseline = calculate_mean_ssim(clean_segment, denoised_baseline, info)
    ssim_spatial = calculate_mean_ssim(clean_segment, denoised_spatial, info)
    
    # Metric 3: Coherence
    print("Calculating Mean Spectral Coherence...")
    coh_baseline = calculate_mean_coherence(clean_segment, denoised_baseline)
    coh_spatial = calculate_mean_coherence(clean_segment, denoised_spatial)

    print("\n--- FINAL RESULTS ---")
    print(f"{'Metric':<25} | {'Baseline (L1)':<15} | {'Spatial STPC':<15}")
    print("-" * 60)
    print(f"{'RMSE':<25} | {rmse_baseline:<15.6f} | {rmse_spatial:<15.6f}")
    print(f"{'Mean SSIM':<25} | {ssim_baseline:<15.6f} | {ssim_spatial:<15.6f}")
    print(f"{'Mean Coherence':<25} | {coh_baseline:<15.6f} | {coh_spatial:<15.6f}")
    print("="*60 + "\n")
    
    # --- Generate Visuals ---
    print("Generating topography video...")
    # ... (the rest of the visualization code remains the same) ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--baseline_model_path', type=str, required=True)
    parser.add_argument('--spatial_model_path', type=str, required=True)
    parser.add_argument('--output_video_path', type=str, required=True)
    cli_args = parser.parse_args()
    validate_and_visualize(cli_args)