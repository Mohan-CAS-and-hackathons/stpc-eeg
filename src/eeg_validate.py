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

from model import UNet1D
from eeg_data_utils import load_eeg_from_edf, TARGET_FS, get_adjacency_list

def calculate_mean_ssim(clean_topo_series, denoised_topo_series, info):
    ssim_scores = []
    fig = plt.figure(figsize=(2, 2), dpi=100)
    num_frames_to_check = 64
    time_indices = np.linspace(0, clean_topo_series.shape[1] - 1, num_frames_to_check).astype(int)
    for i in tqdm(time_indices, desc="Calculating SSIM frames"):
        ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
        vlim = (np.min(clean_topo_series), np.max(clean_topo_series))
        mne.viz.plot_topomap(clean_topo_series[:, i], info, axes=ax1, show=False, vlim=vlim)
        fig.canvas.draw()
        clean_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        mne.viz.plot_topomap(denoised_topo_series[:, i], info, axes=ax2, show=False, vlim=vlim)
        fig.canvas.draw()
        denoised_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        score = ssim(clean_img, denoised_img, channel_axis=-1, data_range=255)
        ssim_scores.append(score)
        fig.clear()
    plt.close(fig)
    return np.mean(ssim_scores)

def calculate_mean_coherence(clean_signal, denoised_signal, fs=TARGET_FS):
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 70)}
    avg_coherence = 0
    for band_name, (fmin, fmax) in bands.items():
        band_coherence_scores = []
        for ch in range(clean_signal.shape[0]):
            f, Cxy = coherence(clean_signal[ch, :], denoised_signal[ch, :], fs=fs, nperseg=fs)
            idx_band = np.where((f >= fmin) & (f <= fmax))
            if len(idx_band[0]) > 0:
                band_coherence_scores.append(np.mean(Cxy[idx_band]))
        if band_coherence_scores:
            avg_coherence += np.mean(band_coherence_scores)
    return avg_coherence / len(bands)

def create_psd_plot(data_dict, fs, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Power Spectral Density Comparison', fontsize=16)
    signals_to_plot = { "Noisy Input": data_dict["Noisy Input"], "Ground Truth": data_dict["Ground Truth"],
                        "Denoised (Frequency STPC)": data_dict["Denoised (Frequency STPC)"]}
    for ax, (title, signal) in zip(axes, signals_to_plot.items()):
        f, Pxx = welch(signal, fs=fs, nperseg=fs*2, axis=-1)
        Pxx_mean = np.mean(Pxx, axis=0)
        ax.semilogy(f, Pxx_mean)
        ax.set_title(title); ax.set_ylabel('Power/Frequency (dB/Hz)'); ax.grid(True, which="both", ls="--")
        ax.axvspan(8, 12, color='orange', alpha=0.2, label='Alpha Band (8-12 Hz)')
        ax.legend(loc='upper right')
    axes[-1].set_xlabel('Frequency (Hz)'); plt.xlim(0, 50); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Saving PSD plot to {output_path}"); plt.savefig(output_path); plt.close(fig)

def create_topomap_video(data_dict, info, output_path, fs):
    print("Generating topography video...")
    vmax = np.max(np.abs(data_dict["Ground Truth"])) * 0.8; vmin = -vmax
    frames = []
    for i in tqdm(range(256), desc="Generating frames"):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'EEG Topography Comparison | Time = {i/fs:.2f} s', fontsize=16)
        for ax, (title, data) in zip(axes, data_dict.items()):
            mne.viz.plot_topomap(data[:, i], info, axes=ax, show=False, vlim=(vmin, vmax))
            ax.set_title(title)
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba()).reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(frame)
        plt.close(fig)
    print(f"Saving video to {output_path}"); imageio.mimsave(output_path, frames, fps=30)

def main(args):
    print(f"--- Starting EEG Denoising Validation for experiment: {args.experiment} ---")
    
    # --- 1. Load Data and Stats ---
    test_file_name = 'chb01_03.edf' if args.experiment == 'spatial' else 'chb01_01.edf'
    test_file_path = os.path.join(args.data_dir, f'chb01/{test_file_name}')
    
    clean_data, final_ch_names = load_eeg_from_edf(test_file_path)
    if clean_data is None: print("Failed to load data."); return
        
    NUM_CHANNELS = clean_data.shape[0]
    
    stats_path = os.path.join(os.path.dirname(args.baseline_model_path), "norm_stats.npz")
    stats = np.load(stats_path)
    mean, std, saved_channels = stats['mean'], stats['std'], stats['channels']
    if not np.array_equal(saved_channels, final_ch_names): print("FATAL: Channel mismatch."); return
    print(f"Data loaded with {NUM_CHANNELS} channels. Normalization stats loaded.")
    
    # --- 2. Load Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(device)
    baseline_model.load_state_dict(torch.load(args.baseline_model_path, map_location=device))
    baseline_model.eval()
    
    # --- 3. Prepare Test Segment and Denoise based on Experiment ---
    if args.experiment == 'spatial':
        start_sample = 2996 * TARGET_FS
        end_sample = start_sample + (4 * TARGET_FS)
        clean_segment = clean_data[:, start_sample:end_sample]
        noise = np.random.randn(*clean_segment.shape).astype(np.float32) * np.std(clean_segment) * 1.5
        noisy_segment = clean_segment + noise
        
        spatial_model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(device)
        spatial_model.load_state_dict(torch.load(args.spatial_model_path, map_location=device))
        spatial_model.eval()

        noisy_norm = (noisy_segment - mean) / (std + 1e-8)
        noisy_tensor = torch.from_numpy(noisy_norm).float().unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_baseline = (baseline_model(noisy_tensor).squeeze(0).cpu().numpy() * (std + 1e-8)) + mean
            denoised_spatial = (spatial_model(noisy_tensor).squeeze(0).cpu().numpy() * (std + 1e-8)) + mean
        
        # --- 4a. Run Spatial Validation ---
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=final_ch_names, sfreq=TARGET_FS, ch_types='eeg')
        info.set_montage(montage, on_missing='ignore')
        
        data_for_video = { "Ground Truth": clean_segment, "Noisy Input": noisy_segment,
                           "Denoised (Baseline L1)": denoised_baseline, "Denoised (Spatial STPC)": denoised_spatial }
        create_topomap_video(data_for_video, info, args.output_path, TARGET_FS)
        
    elif args.experiment == 'frequency':
        start_sample = 1800 * TARGET_FS
        end_sample = start_sample + (4 * TARGET_FS)
        clean_segment = clean_data[:, start_sample:end_sample]
        low_freq = np.sin(2 * np.pi * 1.5 * np.linspace(0, 4, clean_segment.shape[1])) * np.std(clean_segment) * 2
        high_freq = np.random.randn(*clean_segment.shape) * np.std(clean_segment) * 0.5
        noisy_segment = clean_segment + low_freq + high_freq
        
        frequency_model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(device)
        frequency_model.load_state_dict(torch.load(args.frequency_model_path, map_location=device))
        frequency_model.eval()

        noisy_norm = (noisy_segment - mean) / (std + 1e-8)
        noisy_tensor = torch.from_numpy(noisy_norm).float().unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_frequency = (frequency_model(noisy_tensor).squeeze(0).cpu().numpy() * (std + 1e-8)) + mean
            
        # --- 4b. Run Frequency Validation ---
        data_for_plot = { "Noisy Input": noisy_segment, "Ground Truth": clean_segment,
                          "Denoised (Frequency STPC)": denoised_frequency }
        create_psd_plot(data_for_plot, TARGET_FS, args.output_path)

    print("✅ Validation complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, choices=['spatial', 'frequency'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--baseline_model_path', type=str, required=True)
    parser.add_argument('--spatial_model_path', type=str)
    parser.add_argument('--frequency_model_path', type=str)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    
    if args.experiment == 'spatial' and not args.spatial_model_path:
        parser.error("--spatial_model_path is required for the 'spatial' experiment.")
    if args.experiment == 'frequency' and not args.frequency_model_path:
        parser.error("--frequency_model_path is required for the 'frequency' experiment.")
        
    main(args)