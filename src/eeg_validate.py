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
from eeg_data_utils import load_eeg_from_edf, TARGET_FS

# --- Main Validation Function ---
def validate_and_visualize(args):
    print("--- Starting EEG Denoising Validation ---")
    
    # --- 1. Load Data ---
    print(f"Loading and preprocessing test file: {args.test_file}")
    # Find the common channels used during training by inspecting the model name
    # This is a robust way to ensure we match the model's expected input
    subject_dir = os.path.dirname(args.test_file)
    # Note: For simplicity, we are assuming the validation script is run after training
    # and re-uses the same common channels. In a larger project, you'd save this list.
    
    # We need to get the channel names to create the MNE info object for plotting
    raw_info = mne.io.read_raw_edf(args.test_file, preload=False, verbose=False)
    common_channels = [ch.upper() for ch in raw_info.ch_names]
    
    clean_data = load_eeg_from_edf(args.test_file, desired_channels=common_channels)
    if clean_data is None:
        print("Failed to load data. Exiting.")
        return
        
    NUM_CHANNELS = clean_data.shape[0]
    print(f"Data loaded with {NUM_CHANNELS} channels.")

    # --- 2. Prepare a Test Segment ---
    # We'll take a segment from the file that is known to contain seizure activity
    # For chb01_03.edf, a seizure starts around 2996 seconds
    seizure_start_sample = 2996 * TARGET_FS
    segment_length = 4 * TARGET_FS # 4-second window
    
    clean_segment = clean_data[:, seizure_start_sample : seizure_start_sample + segment_length]
    
    # Create a noisy version for testing
    noise = np.random.randn(*clean_segment.shape).astype(np.float32) * np.std(clean_segment) * 1.5
    noisy_segment = clean_segment + noise
    
    # --- 3. Load Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    baseline_model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(device)
    baseline_model.load_state_dict(torch.load(args.baseline_model_path, map_location=device))
    baseline_model.eval()
    print("Baseline model loaded.")

    spatial_model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(device)
    spatial_model.load_state_dict(torch.load(args.spatial_model_path, map_location=device))
    spatial_model.eval()
    print("Spatial model loaded.")
    
    # --- 4. Denoise the Segment with Both Models ---
    noisy_tensor = torch.from_numpy(noisy_segment).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_baseline = baseline_model(noisy_tensor).squeeze(0).cpu().numpy()
        denoised_spatial = spatial_model(noisy_tensor).squeeze(0).cpu().numpy()
    print("Denoising complete.")
    
    # --- 5. Generate the Knockout Visual (Video) ---
    print("Generating topography video... this may take a few minutes.")
    
    # Create an MNE Info object needed for plotting topomaps
    # We use a standard 1020 montage which MNE provides
    ch_types = ['eeg'] * NUM_CHANNELS
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=common_channels, sfreq=TARGET_FS, ch_types=ch_types)
    info.set_montage(montage, on_missing='ignore')

    data_to_plot = {
        "Ground Truth": clean_segment,
        "Noisy Input": noisy_segment,
        "Denoised (Baseline L1)": denoised_baseline,
        "Denoised (Spatial STPC)": denoised_spatial,
    }

    # Find a common color scale for all plots
    vmax = np.max(np.abs(clean_segment))
    vmin = -vmax

    frames = []
    # We'll create a video of the first 2 seconds (512 frames)
    for i in tqdm(range(512), desc="Generating frames"):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'EEG Topography Comparison | Time = {i/TARGET_FS:.2f} s', fontsize=16)
        
        for ax, (title, data) in zip(axes, data_to_plot.items()):
            mne.viz.plot_topomap(data[:, i], info, axes=ax, show=False, vlim=(vmin, vmax))
            ax.set_title(title, fontsize=12)
        
        # Save figure to a numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)
        
    # Save frames as a GIF
    print(f"Saving video to {args.output_video_path}")
    imageio.mimsave(args.output_video_path, frames, fps=30)
    print("✅ Validation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate and visualize EEG denoiser performance.")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the .edf file for testing (e.g., chb01_03.edf).")
    parser.add_argument('--baseline_model_path', type=str, required=True, help="Path to the trained baseline model.")
    parser.add_argument('--spatial_model_path', type=str, required=True, help="Path to the trained spatial STPC model.")
    parser.add_argument('--output_video_path', type=str, required=True, help="Path to save the output comparison video (e.g., comparison.gif).")
    
    cli_args = parser.parse_args()
    validate_and_visualize(cli_args)