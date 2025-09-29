# src/experiments/run_validation.py
import os
import sys
import argparse
import torch
import numpy as np
import wfdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import mne
import imageio
from scipy.signal import welch
import umap.umap_ as umap
from sklearn.metrics import silhouette_score

# ==============================================================================
#                      SETUP: Add project root to system path
# ==============================================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==============================================================================

from src.stpc.model import UNet1D, ECGClassifier
from src.stpc.utils.ecg_utils import TARGET_FS as ECG_FS, BEAT_CLASSES, BEAT_WINDOW_SIZE, get_noise_signals, load_and_resample_signal as load_ecg_signal, working_directory
from src.stpc.utils.eeg_utils import load_eeg_from_edf, create_eeg_segments, TARGET_FS as EEG_FS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
#                          HELPER FUNCTIONS
# ==============================================================================

def create_topomap_video(data_dict, info, output_path):
    print("Generating topography video...")
    vmax = np.max(np.abs(data_dict["Ground Truth"])) * 0.8; vmin = -vmax; frames = []
    for i in tqdm(range(256), desc="Generating frames for video"):
        fig, axes = plt.subplots(1, len(data_dict), figsize=(5 * len(data_dict), 5))
        if len(data_dict) == 1: axes = [axes]
        fig.suptitle(f'EEG Topography Comparison | Time = {i/EEG_FS:.2f} s', fontsize=16)
        for ax, (title, data) in zip(axes, data_dict.items()):
            mne.viz.plot_topomap(data[:, i], info, axes=ax, show=False, vlim=(vmin, vmax))
            ax.set_title(title)
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba()).reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(frame); plt.close(fig)
    print(f"Saving video to {output_path}"); imageio.mimsave(output_path, frames, fps=30, macro_block_size=1)

def create_psd_plot(data_dict, output_path):
    print("Generating PSD plot...")
    fig, axes = plt.subplots(len(data_dict), 1, figsize=(12, 4 * len(data_dict)), sharex=True)
    if len(data_dict) == 1: axes = [axes]
    fig.suptitle('Power Spectral Density Comparison', fontsize=16)
    for ax, (title, signal) in zip(axes, data_dict.items()):
        f, Pxx = welch(signal, fs=EEG_FS, nperseg=EEG_FS*2, axis=-1); Pxx_mean = np.mean(Pxx, axis=0)
        ax.semilogy(f, Pxx_mean); ax.set_title(title); ax.set_ylabel('Power/Frequency (dB/Hz)'); ax.grid(True, which="both", ls="--")
        ax.axvspan(8, 12, color='orange', alpha=0.2, label='Alpha Band (8-12 Hz)'); ax.legend(loc='upper right')
    axes[-1].set_xlabel('Frequency (Hz)'); plt.xlim(0, 50); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Saving PSD plot to {output_path}"); plt.savefig(output_path); plt.close(fig)

def create_embedding_plot(model, data_loader, output_path):
    all_embeddings, all_labels = [], []
    print("Extracting embeddings...");
    for segments, labels in tqdm(data_loader):
        segments = segments.to(DEVICE)
        with torch.no_grad():
            embeddings = model.encode(segments).cpu().numpy()
        all_embeddings.append(embeddings); all_labels.append(labels.numpy())
    all_embeddings, all_labels = np.concatenate(all_embeddings), np.concatenate(all_labels)
    print("Running UMAP...");
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)
    score = silhouette_score(embedding_2d, all_labels)
    print("Generating scatter plot...");
    plt.figure(figsize=(10, 8)); scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=all_labels, cmap='viridis', s=10, alpha=0.7)
    plt.title(f'UMAP Projection of Learned EEG Embeddings\nSilhouette Score: {score:.4f}', fontsize=16)
    plt.xlabel("UMAP Dimension 1"); plt.ylabel("UMAP Dimension 2"); plt.legend(handles=scatter.legend_elements()[0], labels=['Non-Seizure', 'Seizure']); plt.grid(True, linestyle='--', alpha=0.6)
    print(f"Saving embedding plot to {output_path}"); plt.savefig(output_path); plt.close()

# ==============================================================================
#                          MAIN VALIDATION FUNCTIONS
# ==============================================================================

def validate_ecg_downstream(args):
    print("--- Running End-to-End ECG Downstream Validation ---")
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    denoiser = UNet1D(in_channels=1, out_channels=1).to(DEVICE)
    denoiser.load_state_dict(torch.load(args.denoiser_path, map_location=DEVICE)); denoiser.eval()
    classifier = ECGClassifier().to(DEVICE)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=DEVICE)); classifier.eval()

    with working_directory(args.data_dir):
        clean_signal = load_ecg_signal(args.record_name, target_fs=ECG_FS)
        annotation = wfdb.rdann(args.record_name, 'atr')
    true_samples = (annotation.sample * (ECG_FS / annotation.fs)).astype('int64')

    noise_signals = get_noise_signals(args.noise_dir, target_fs=ECG_FS)
    noise = np.tile(noise_signals['muscle_artifact'], int(np.ceil(len(clean_signal)/len(noise_signals['muscle_artifact']))))[:len(clean_signal)]
    power_clean = np.mean(clean_signal**2)
    scaling = np.sqrt((power_clean / (10**(args.snr_db / 10))) / np.mean(noise**2))
    noisy_signal = clean_signal + noise * scaling

    denoised_signal = np.zeros_like(noisy_signal)
    for i in tqdm(range(0, len(noisy_signal), 2048), desc="Denoising"):
        segment = noisy_signal[i:i+2048]
        if len(segment) < 2048: segment = np.pad(segment, (0, 2048-len(segment)))
        with torch.no_grad():
            tensor_in = torch.from_numpy(segment.copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            tensor_out = denoiser(tensor_in).squeeze().cpu().numpy()
        denoised_signal[i:i+len(noisy_signal[i:i+2048])] = tensor_out[:len(noisy_signal[i:i+2048])]
        
    for name, sig in {'Noisy': noisy_signal, 'Denoised': denoised_signal, 'Clean': clean_signal}.items():
        preds, truth = [], []
        for i, sym in enumerate(annotation.symbol):
            if sym in BEAT_CLASSES:
                loc = true_samples[i]
                start, end = loc - BEAT_WINDOW_SIZE//2, loc + BEAT_WINDOW_SIZE//2
                if start >= 0 and end < len(sig):
                    with torch.no_grad():
                        tensor_in = torch.from_numpy(sig[start:end].copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                        preds.append(torch.argmax(classifier(tensor_in), dim=1).item())
                    truth.append(BEAT_CLASSES[sym])
        
        print(f"\n--- PERFORMANCE ON {name.upper()} SIGNAL ---")
        print(classification_report(truth, preds, target_names=['N','S','V','F','Q'], zero_division=0))
        cm = confusion_matrix(truth, preds, labels=range(5))
        disp = ConfusionMatrixDisplay(cm, display_labels=['N','S','V','F','Q'])
        disp.plot(cmap=plt.cm.Blues).ax_.set_title(f'CM - {name} Signal')
        plt.savefig(f'{args.output_prefix}_cm_{name.lower()}.png'); plt.close()

def validate_eeg(args):
    print(f"--- Running EEG Validation: {args.eeg_experiment_type} ---")
    
    if args.eeg_experiment_type == 'spatial':
        # DEFINITIVE FIX: Use the info object returned by the data loader
        clean_data, final_ch_names, info = load_eeg_from_edf(args.test_file_path)
        if info is None: raise RuntimeError("Failed to load data or create MNE info object.")
        NUM_CHANNELS = len(final_ch_names)
        
        stats = np.load(os.path.join(os.path.dirname(args.baseline_model_path), "norm_stats.npz"))
        mean, std = stats['mean'], stats['std']

        baseline_model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(DEVICE)
        baseline_model.load_state_dict(torch.load(args.baseline_model_path, map_location=DEVICE)); baseline_model.eval()
        
        spatial_model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(DEVICE)
        spatial_model.load_state_dict(torch.load(args.spatial_model_path, map_location=DEVICE)); spatial_model.eval()

        start = 2996 * EEG_FS; end = start + (4 * EEG_FS)
        clean_segment = clean_data[:, start:end]
        noise = np.random.randn(*clean_segment.shape).astype(np.float32) * np.std(clean_segment) * 1.5
        noisy_segment = clean_segment + noise
        
        noisy_norm = (noisy_segment - mean) / (std + 1e-8)
        noisy_tensor = torch.from_numpy(noisy_norm).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            denoised_baseline = (baseline_model(noisy_tensor).squeeze(0).cpu().numpy() * (std + 1e-8)) + mean
            denoised_spatial = (spatial_model(noisy_tensor).squeeze(0).cpu().numpy() * (std + 1e-8)) + mean
            
        data_for_video = {"Ground Truth": clean_segment, "Noisy Input": noisy_segment,
                          "Denoised (Baseline L1)": denoised_baseline, "Denoised (Spatial STPC)": denoised_spatial}
        # Pass the correctly loaded and configured 'info' object
        create_topomap_video(data_for_video, info, args.output_path)


    elif args.eeg_experiment_type == 'frequency':
        clean_data, final_ch_names, _ = load_eeg_from_edf(args.test_file_path)
        NUM_CHANNELS = len(final_ch_names)
        stats = np.load(os.path.join(os.path.dirname(args.frequency_model_path), "norm_stats.npz"))
        mean, std = stats['mean'], stats['std']
        frequency_model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(DEVICE)
        frequency_model.load_state_dict(torch.load(args.frequency_model_path, map_location=DEVICE)); frequency_model.eval()
        start = 1800 * EEG_FS; end = start + (4 * EEG_FS)
        clean_segment = clean_data[:, start:end]
        low_freq = np.sin(2*np.pi*1.5*np.linspace(0, 4, clean_segment.shape[1]))*np.std(clean_segment)*2
        high_freq = np.random.randn(*clean_segment.shape)*np.std(clean_segment)*0.5
        noisy_segment = clean_segment + low_freq + high_freq
        noisy_norm = (noisy_segment - mean) / (std + 1e-8)
        noisy_tensor = torch.from_numpy(noisy_norm).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            denoised_frequency = (frequency_model(noisy_tensor).squeeze(0).cpu().numpy() * (std + 1e-8)) + mean
        data_for_plot = {"Noisy Input": noisy_segment, "Ground Truth": clean_segment,
                         "Denoised (Frequency STPC)": denoised_frequency}
        create_psd_plot(data_for_plot, args.output_path)

    elif args.eeg_experiment_type == 'self_supervised':
        seizure_data, _, _ = load_eeg_from_edf(os.path.join(args.data_dir, 'chb01/chb01_03.edf'))
        non_seizure_data, ch_names, _ = load_eeg_from_edf(os.path.join(args.data_dir, 'chb01/chb01_01.edf'))
        NUM_CHANNELS = len(ch_names)
        seizure_segs = create_eeg_segments(seizure_data, 2 * EEG_FS)
        non_seizure_segs = create_eeg_segments(non_seizure_data, 2 * EEG_FS)
        test_segments = np.array(non_seizure_segs[850:1050] + seizure_segs[1400:1600])
        test_labels = np.array([0]*200 + [1]*200)
        model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(DEVICE)
        model.load_state_dict(torch.load(args.self_supervised_model_path, map_location=DEVICE)); model.eval()
        stats = np.load(os.path.join(os.path.dirname(args.self_supervised_model_path), "norm_stats.npz"))
        mean, std = stats['mean'], stats['std']
        test_segments_norm = (test_segments - mean) / (std + 1e-8)
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_segments_norm).float(), torch.from_numpy(test_labels))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        create_embedding_plot(model, test_loader, args.output_path)

# ==============================================================================
#                               MAIN SCRIPT LOGIC
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Validation Runner")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    p_ecg = subparsers.add_parser("ecg_downstream", help="Run ECG downstream classification validation.")
    p_ecg.add_argument("--denoiser_path", type=str, required=True)
    p_ecg.add_argument("--classifier_path", type=str, required=True)
    p_ecg.add_argument("--data_dir", type=str, required=True)
    p_ecg.add_argument("--noise_dir", type=str, required=True)
    p_ecg.add_argument("--output_prefix", type=str, required=True)
    p_ecg.add_argument("--record_name", type=str, default="201")
    p_ecg.add_argument("--snr_db", type=int, default=0)

    p_eeg = subparsers.add_parser("eeg", help="Run EEG validation tasks.")
    p_eeg.add_argument("--eeg_experiment_type", type=str, required=True,
                       choices=['spatial', 'frequency', 'self_supervised'])
    p_eeg.add_argument("--data_dir", type=str, required=True)
    p_eeg.add_argument("--output_path", type=str, required=True)
    p_eeg.add_argument("--baseline_model_path", type=str)
    p_eeg.add_argument("--spatial_model_path", type=str)
    p_eeg.add_argument("--frequency_model_path", type=str)
    p_eeg.add_argument("--self_supervised_model_path", type=str)
    p_eeg.add_argument("--test_file_path", type=str)
    
    args = parser.parse_args()

    if args.experiment == "ecg_downstream":
        validate_ecg_downstream(args)
    elif args.experiment == "eeg":
        if args.eeg_experiment_type == 'spatial':
            if not all([args.baseline_model_path, args.spatial_model_path, args.test_file_path]):
                parser.error("Spatial experiment requires --baseline_model_path, --spatial_model_path, and --test_file_path.")
        elif args.eeg_experiment_type == 'frequency':
            if not all([args.frequency_model_path, args.test_file_path]):
                parser.error("Frequency experiment requires --frequency_model_path and --test_file_path.")
        elif args.eeg_experiment_type == 'self_supervised':
            if not args.self_supervised_model_path:
                parser.error("--self_supervised_model_path is required.")
        
        # This is the corrected call
        validate_eeg(args)