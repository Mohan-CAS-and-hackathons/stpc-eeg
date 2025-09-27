# src/eeg_validate.py
import torch
import numpy as np
import os
import argparse
import mne
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt

from model import UNet1D
from eeg_data_utils import load_eeg_from_edf, TARGET_FS # Only need these two now

def validate_and_visualize(args):
    print("--- Starting EEG Denoising Validation ---")
    
    print(f"Loading and preprocessing test file: {args.test_file}")
    
    # --- KEY FIX: Load data AND the new, clean monopolar channel names ---
    clean_data, final_ch_names = load_eeg_from_edf(args.test_file)
    if clean_data is None:
        print("Failed to load data. Exiting.")
        return
        
    NUM_CHANNELS = clean_data.shape[0]
    print(f"Data loaded with {NUM_CHANNELS} unique monopolar channels.")

    seizure_start_sample = 2996 * TARGET_FS
    segment_length = 4 * TARGET_FS
    clean_segment = clean_data[:, seizure_start_sample : seizure_start_sample + segment_length]
    noise = np.random.randn(*clean_segment.shape).astype(np.float32) * np.std(clean_segment) * 1.5
    noisy_segment = clean_segment + noise
    
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

    # Note: We must now train models on the unique monopolar data.
    # We will need to re-run training one last time.
    baseline_model = load_model(args.baseline_model_path, NUM_CHANNELS, NUM_CHANNELS, device)
    spatial_model = load_model(args.spatial_model_path, NUM_CHANNELS, NUM_CHANNELS, device)
    
    noisy_tensor = torch.from_numpy(noisy_segment).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_baseline = baseline_model(noisy_tensor).squeeze(0).cpu().numpy() if baseline_model else np.zeros_like(noisy_segment)
        denoised_spatial = spatial_model(noisy_tensor).squeeze(0).cpu().numpy() if spatial_model else np.zeros_like(noisy_segment)
    print("Denoising complete.")
    
    print("Generating topography video...")
    
    # --- KEY FIX: Creating the info object is now simple and direct ---
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=final_ch_names, sfreq=TARGET_FS, ch_types='eeg')
    info.set_montage(montage, on_missing='ignore') # 'ignore' handles any non-standard names
    # --- END FIX ---
    
    data_to_plot = {"Ground Truth": clean_segment, "Noisy Input": noisy_segment,
                    "Denoised (Baseline L1)": denoised_baseline, "Denoised (Spatial STPC)": denoised_spatial}
    vmax = np.max(np.abs(clean_segment))
    vmin = -vmax

    frames = []
    for i in tqdm(range(256), desc="Generating frames"):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'EEG Topography Comparison | Time = {i/TARGET_FS:.2f} s', fontsize=16)
        
        for ax, (title, data) in zip(axes, data_to_plot.items()):
            # The data and info object are now perfectly aligned
            mne.viz.plot_topomap(data[:, i], info, axes=ax, show=False, vlim=(vmin, vmax))
            ax.set_title(title, fontsize=12)
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)
        
    print(f"Saving video to {args.output_video_path}")
    imageio.mimsave(args.output_video_path, frames, fps=30)
    print("✅ Validation complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--baseline_model_path', type=str, required=True)
    parser.add_argument('--spatial_model_path', type=str, required=True)
    parser.add_argument('--output_video_path', type=str, required=True)
    cli_args = parser.parse_args()
    validate_and_visualize(cli_args)