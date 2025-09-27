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

def validate_and_visualize(args):
    print("--- Starting EEG Denoising Validation ---")
    
    # --- 1. Load Data ---
    print(f"Loading and preprocessing test file: {args.test_file}")
    
    # Get the raw channel names from the file to build the info object
    raw_info_obj = mne.io.read_raw_edf(args.test_file, preload=False, verbose=False)
    common_channels = [ch.upper() for ch in raw_info_obj.ch_names]
    
    clean_data = load_eeg_from_edf(args.test_file, desired_channels=common_channels)
    if clean_data is None:
        print("Failed to load data. Exiting.")
        return
        
    NUM_CHANNELS = clean_data.shape[0]
    print(f"Data loaded with {NUM_CHANNELS} channels.")

    # --- 2. Prepare a Test Segment ---
    seizure_start_sample = 2996 * TARGET_FS
    segment_length = 4 * TARGET_FS
    clean_segment = clean_data[:, seizure_start_sample : seizure_start_sample + segment_length]
    noise = np.random.randn(*clean_segment.shape).astype(np.float32) * np.std(clean_segment) * 1.5
    noisy_segment = clean_segment + noise
    
    # --- 3. Load Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use a dummy model if a path is 'None'
    def load_model(path, in_ch, out_ch, dev):
        model = UNet1D(in_channels=in_ch, out_channels=out_ch).to(dev)
        if path and os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=dev))
            print(f"Model loaded from {path}.")
        else:
            print(f"Warning: Model not found at {path}. Using dummy model.")
        model.eval()
        return model

    baseline_model = load_model(args.baseline_model_path, NUM_CHANNELS, NUM_CHANNELS, device)
    spatial_model = load_model(args.spatial_model_path, NUM_CHANNELS, NUM_CHANNELS, device)
    
    # --- 4. Denoise the Segment with Both Models ---
    noisy_tensor = torch.from_numpy(noisy_segment).unsqueeze(0).to(device)
    with torch.no_grad():
        denoised_baseline = baseline_model(noisy_tensor).squeeze(0).cpu().numpy()
        denoised_spatial = spatial_model(noisy_tensor).squeeze(0).cpu().numpy()
    print("Denoising complete.")
    
    # --- 5. Generate the Knockout Visual (Video) ---
    print("Generating topography video... this may take a few minutes.")
    
    # --- KEY MODIFICATION: Manually setting the montage ---
    # We create a standard 10-20 montage which contains the 3D electrode locations.
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Some channel names in the data (like T8-P8-1) don't exist in the standard montage.
    # We'll create a new list of channel names that MNE can find in the montage.
    info_ch_names = []
    data_indices_to_plot = []
    for i, ch in enumerate(common_channels):
        # MNE automatically handles standard bipolar names like 'FP1-F7'.
        # We need to clean up the duplicate names like 'T8-P8-1' -> 'T8-P8'.
        cleaned_ch = ch.split('-')[0] + '-' + ch.split('-')[1].split(' ')[0]
        if cleaned_ch in montage.ch_names:
            info_ch_names.append(ch) # Keep original name for the info object
            data_indices_to_plot.append(i) # Keep track of which row of data to use
    
    # Create the MNE Info object with the channels that have known locations
    info = mne.create_info(ch_names=info_ch_names, sfreq=TARGET_FS, ch_types='eeg')
    info.set_montage(montage)
    # ----------------------------------------------------------------------
    
    data_to_plot = {
        "Ground Truth": clean_segment,
        "Noisy Input": noisy_segment,
        "Denoised (Baseline L1)": denoised_baseline,
        "Denoised (Spatial STPC)": denoised_spatial,
    }

    vmax = np.max(np.abs(clean_segment))
    vmin = -vmax

    frames = []
    for i in tqdm(range(512), desc="Generating frames"):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'EEG Topography Comparison | Time = {i/TARGET_FS:.2f} s', fontsize=16)
        
        for ax, (title, data) in zip(axes, data_to_plot.items()):
            # Select only the data from channels that are in our info object
            data_for_topo = data[data_indices_to_plot, i]
            mne.viz.plot_topomap(data_for_topo, info, axes=ax, show=False, vlim=(vmin, vmax))
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
    parser = argparse.ArgumentParser(description="Validate and visualize EEG denoiser performance.")
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--baseline_model_path', type=str, required=True)
    parser.add_argument('--spatial_model_path', type=str, required=True)
    parser.add_argument('--output_video_path', type=str, required=True)
    
    cli_args = parser.parse_args()
    validate_and_visualize(cli_args)