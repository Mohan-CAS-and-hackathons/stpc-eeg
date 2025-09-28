# src/experiments/run_validation.py
import torch
import numpy as np
import os
import argparse
import mne
import imageio
import matplotlib.pyplot as plt
import wfdb
from tqdm import tqdm
from scipy.signal import welch
import umap.umap_ as umap
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, silhouette_score

# --- Import from our new STPC library ---
from stpc.model import UNet1D, ECGClassifier
from stpc.utils.ecg_utils import TARGET_FS as ECG_TARGET_FS, BEAT_CLASSES, BEAT_WINDOW_SIZE, get_noise_signals
from stpc.utils.eeg_utils import TARGET_FS as EEG_TARGET_FS, load_eeg_from_edf, create_eeg_segments

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
#                      VALIDATION LOGIC FOR ECG DOWNSTREAM
# ==============================================================================
def validate_ecg_downstream(args):
    """
    Full end-to-end validation for the ECG denoiser's impact on a downstream classifier.
    """
    print("--- Running End-to-End ECG Downstream Validation ---")
    print(f"Test Record: {args.record_name}, Noise Level: {args.snr_db} dB SNR")
    print(f"Loading Denoiser from: {args.denoiser_path}")
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    # --- Load Models ---
    print("Loading models...")
    denoiser = UNet1D().to(DEVICE); denoiser.load_state_dict(torch.load(args.denoiser_path, map_location=DEVICE)); denoiser.eval()
    classifier = ECGClassifier().to(DEVICE); classifier.load_state_dict(torch.load(args.classifier_path, map_location=DEVICE)); classifier.eval()

    # --- Load Data ---
    print(f"Downloading record '{args.record_name}' from PhysioNet...")
    wfdb.dl_database('mitdb', '.', records=[args.record_name])
    record = wfdb.rdrecord(args.record_name)
    annotation = wfdb.rdann(args.record_name, 'atr')
    clean_signal = wfdb.processing.resample_sig(record.p_signal[:, 0], record.fs, ECG_TARGET_FS)[0]
    true_samples = (annotation.sample * (ECG_TARGET_FS / record.fs)).astype('int64')

    # --- Synthesize Noise ---
    print("Synthesizing noisy signal...")
    noise_signals = get_noise_signals(args.noise_dir, ECG_TARGET_FS)
    noise_type = 'muscle_artifact' # A challenging default
    if noise_type not in noise_signals:
        noise_type = list(noise_signals.keys())[0] # Fallback to first available
    
    long_noise = np.tile(noise_signals[noise_type], int(np.ceil(len(clean_signal) / len(noise_signals[noise_type]))))
    long_noise = long_noise[:len(clean_signal)]
    
    power_clean = np.mean(clean_signal ** 2)
    power_noise = np.mean(long_noise ** 2)
    scaling_factor = np.sqrt((power_clean / (10**(args.snr_db / 10))) / power_noise) if power_noise > 0 else 0
    noisy_signal = clean_signal + long_noise * scaling_factor

    # --- Denoise Signal ---
    print("Denoising signal...")
    segment_length = 2048
    denoised_signal = np.zeros_like(noisy_signal)
    for i in tqdm(range(0, len(noisy_signal), segment_length), desc="Denoising"):
        segment = noisy_signal[i:i+segment_length]
        original_len = len(segment)
        if original_len < segment_length:
            segment = np.pad(segment, (0, segment_length - original_len), 'constant')

        with torch.no_grad():
            tensor_in = torch.from_numpy(segment.copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            tensor_out = denoiser(tensor_in).squeeze().cpu().numpy()
        
        denoised_signal[i:i+segment_length] = tensor_out[:original_len]
        
    # --- Classify and Evaluate ---
    signals_to_test = {'Noisy': noisy_signal, 'Denoised': denoised_signal, 'Clean': clean_signal}
    final_class_names = ['N', 'S', 'V', 'F', 'Q']
    
    for name, sig in signals_to_test.items():
        predictions, ground_truth = [], []
        for i, sym in enumerate(annotation.symbol):
            if sym in BEAT_CLASSES:
                loc = true_samples[i]
                start, end = loc - BEAT_WINDOW_SIZE//2, loc + BEAT_WINDOW_SIZE//2
                if start >= 0 and end < len(sig):
                    beat_window = sig[start:end]
                    with torch.no_grad():
                        tensor_in = torch.from_numpy(beat_window.copy()).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                        pred_label = torch.argmax(classifier(tensor_in), dim=1).item()
                    predictions.append(pred_label)
                    ground_truth.append(BEAT_CLASSES[sym])
        
        print(f"\n--- PERFORMANCE ON {name.upper()} SIGNAL ---")
        print(classification_report(ground_truth, predictions, target_names=final_class_names, zero_division=0))
        cm = confusion_matrix(ground_truth, predictions, labels=range(len(final_class_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_class_names)
        disp.plot(cmap=plt.cm.Blues)
        disp.ax_.set_title(f'Confusion Matrix - {name} Signal')
        plt.savefig(f'{args.output_prefix}_cm_{name.lower()}.png', bbox_inches='tight')
        plt.close()

# ==============================================================================
#                      VALIDATION LOGIC FOR EEG EXPERIMENTS
# ==============================================================================
# --- Helper Plotting Functions ---
def create_topomap_video(data_dict, info, output_path, fs):
    print("Generating topography video..."); frames = []
    vmax = np.max(np.abs(data_dict["Ground Truth"])) * 0.8; vmin = -vmax
    for i in tqdm(range(0, info.get('n_times', 256)), desc="Generating frames"):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5)); fig.suptitle(f'EEG Topography Comparison | Time = {i/fs:.2f} s', fontsize=16)
        for ax, (title, data) in zip(axes, data_dict.items()):
            mne.viz.plot_topomap(data[:, i], info, axes=ax, show=False, vlim=(vmin, vmax))
            ax.set_title(title)
        fig.canvas.draw(); frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,)); frames.append(frame); plt.close(fig)
    print(f"Saving video to {output_path}"); imageio.mimsave(output_path, frames, fps=30)

def create_psd_plot(data_dict, fs, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True); fig.suptitle('Power Spectral Density Comparison', fontsize=16)
    signals_to_plot = { "Noisy Input": data_dict["Noisy Input"], "Ground Truth": data_dict["Ground Truth"], "Denoised (Frequency STPC)": data_dict["Denoised (Frequency STPC)"]}
    for ax, (title, signal) in zip(axes, signals_to_plot.items()):
        f, Pxx = welch(signal, fs=fs, nperseg=fs*2, axis=-1); Pxx_mean = np.mean(Pxx, axis=0)
        ax.semilogy(f, Pxx_mean); ax.set_title(title); ax.set_ylabel('Power/Frequency (dB/Hz)'); ax.grid(True, which="both", ls="--")
        ax.axvspan(8, 12, color='orange', alpha=0.2, label='Alpha Band (8-12 Hz)'); ax.legend(loc='upper right')
    axes[-1].set_xlabel('Frequency (Hz)'); plt.xlim(0, 50); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Saving PSD plot to {output_path}"); plt.savefig(output_path); plt.close(fig)

def create_embedding_plot(model, data_loader, device, output_path):
    all_embeddings, all_labels = [], []
    print("Extracting embeddings from test data...")
    for segments, labels in tqdm(data_loader):
        segments = segments.to(device)
        with torch.no_grad():
            embeddings = model.encode(segments).cpu().numpy()
        all_embeddings.append(embeddings); all_labels.append(labels.numpy())
        
    all_embeddings = np.concatenate(all_embeddings); all_labels = np.concatenate(all_labels)

    print("Running UMAP...")
    embedding_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(all_embeddings)
    score = silhouette_score(embedding_2d, all_labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=all_labels, cmap='viridis', s=10, alpha=0.7)
    plt.title(f'UMAP Projection of Learned EEG Embeddings\nSilhouette Score: {score:.4f}', fontsize=16)
    plt.xlabel("UMAP Dimension 1"); plt.ylabel("UMAP Dimension 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=['Non-Seizure', 'Seizure'])
    plt.grid(True, linestyle='--', alpha=0.6); plt.savefig(output_path); plt.close()
    print(f"Saving embedding plot to {output_path}")

# --- Main EEG Validation Function ---
def validate_eeg(args):
    print(f"--- Running EEG Validation for: {args.validation_task} ---")
    
    # Common data loading
    gt_data, ch_names, info = load_eeg_from_edf(args.test_file_path)
    if gt_data is None: raise ValueError("Could not load test EEG file.")
    NUM_CHANNELS = len(ch_names)
    
    # Denoise with a specified model
    def denoise_eeg(model_path, noisy_data, num_channels):
        model = UNet1D(in_channels=num_channels, out_channels=num_channels).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE)); model.eval()
        
        # Load stats
        stats = np.load(os.path.join(os.path.dirname(model_path), "norm_stats.npz"))
        mean, std = stats['mean'], stats['std']
        
        noisy_norm = (noisy_data - mean) / (std + 1e-8)
        denoised_norm = np.zeros_like(noisy_norm)
        
        # Process in segments
        segments = create_eeg_segments(noisy_norm, 2 * EEG_TARGET_FS)
        denoised_segments = []
        with torch.no_grad():
            for seg in segments:
                tensor_in = torch.from_numpy(seg).float().unsqueeze(0).to(DEVICE)
                out = model(tensor_in).squeeze().cpu().numpy()
                denoised_segments.append(out)
        
        # Reconstruct the signal
        # (Simplified reconstruction for plotting; overlap-add would be more robust)
        denoised_norm = np.concatenate(denoised_segments, axis=1)
        denoised_signal = (denoised_norm * std) + mean
        return denoised_signal[:, :noisy_data.shape[1]]

    # --- Task-specific logic ---
    if args.validation_task == "eeg_spatiotemporal":
        noisy_data = gt_data + np.random.randn(*gt_data.shape) * np.std(gt_data) * 1.5
        denoised_base = denoise_eeg(args.baseline_model_path, noisy_data, NUM_CHANNELS)
        denoised_spatial = denoise_eeg(args.spatial_model_path, noisy_data, NUM_CHANNELS)
        
        data_dict = {"Ground Truth": gt_data, "Noisy Input": noisy_data,
                     "Denoised (Baseline L1)": denoised_base, "Denoised (Spatial STPC)": denoised_spatial}
        info['n_times'] = gt_data.shape[1] # Ensure info object is correct
        create_topomap_video(data_dict, info, args.output_path, EEG_TARGET_FS)

    elif args.validation_task == "eeg_frequency":
        print("Validating Frequency-Specific Preservation (PSD Plot)")
        
        # --- Helper to create a specific type of noisy signal for this test ---
        def create_frequency_test_signal(base_signal):
            """Adds a strong low-frequency drift and some high-frequency noise."""
            # Add strong baseline wander (e.g., at 1 Hz)
            time = np.arange(base_signal.shape[1]) / EEG_TARGET_FS
            low_freq_noise = np.sin(2 * np.pi * 1.0 * time) * np.std(base_signal) * 2.0
            
            # Add some broadband white noise
            white_noise = np.random.randn(*base_signal.shape) * np.std(base_signal) * 0.5
            
            return base_signal + low_freq_noise[np.newaxis, :] + white_noise

        # --- Main Logic ---
        # We need a ground truth signal that has some alpha-band activity.
        # Let's load a segment from a file known to have resting-state alpha waves.
        # We'll use chb01_01.edf as a stand-in for this example.
        alpha_test_file = os.path.join(args.data_dir, 'chb01/chb01_01.edf')
        gt_data, ch_names, info = load_eeg_from_edf(alpha_test_file)
        if gt_data is None: raise ValueError("Could not load EEG file for frequency test.")
        
        # We only need a few seconds for a PSD plot
        gt_segment = gt_data[:, :EEG_TARGET_FS * 10] # 10 seconds
        NUM_CHANNELS = len(ch_names)

        # Create the specific noisy signal for this experiment
        noisy_segment = create_frequency_test_signal(gt_segment)

        # Denoise using the frequency-tuned model
        denoised_segment = denoise_eeg(args.frequency_model_path, noisy_segment, NUM_CHANNELS)
        
        # We will average the PSD across all channels for a cleaner plot
        data_dict = {
            "Noisy Input": np.mean(noisy_segment, axis=0), 
            "Ground Truth": np.mean(gt_segment, axis=0),
            "Denoised (Frequency STPC)": np.mean(denoised_segment, axis=0)
        }
        
        create_psd_plot(data_dict, EEG_TARGET_FS, args.output_path)
        
    elif args.validation_task == "eeg_unsupervised":
        model = UNet1D(in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS).to(DEVICE)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE)); model.eval()
        
        # Create a test dataset of seizure vs non-seizure
        seizure_file = os.path.join(args.data_dir, 'chb01/chb01_03.edf')
        non_seizure_file = os.path.join(args.data_dir, 'chb01/chb01_01.edf')
        seizure_data, _, _ = load_eeg_from_edf(seizure_file)
        non_seizure_data, _, _ = load_eeg_from_edf(non_seizure_file)
        
        seizure_segments = create_eeg_segments(seizure_data, 2 * EEG_TARGET_FS)
        non_seizure_segments = create_eeg_segments(non_seizure_data, 2 * EEG_TARGET_FS)
        
        test_segments = np.array(non_seizure_segments[:200] + seizure_segments[1400:1600])
        test_labels = np.array([0]*200 + [1]*200)

        stats = np.load(os.path.join(os.path.dirname(args.model_path), "norm_stats.npz"))
        mean, std = stats['mean'], stats['std']
        test_segments_norm = (test_segments - mean[:, :, np.newaxis]) / (std[:, :, np.newaxis] + 1e-8)
        
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_segments_norm).float(), torch.from_numpy(test_labels))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        
        create_embedding_plot(model, test_loader, DEVICE, args.output_path)

# ==============================================================================
#                               MAIN SCRIPT LOGIC
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Validation Runner for STPC Experiments")
    subparsers = parser.add_subparsers(dest="validation_task", required=True, help="The validation task to run")

    # --- ECG Subparser ---
    p_ecg = subparsers.add_parser("ecg_downstream", help="Run downstream classification on a denoised ECG signal.")
    p_ecg.add_argument("--denoiser_path", type=str, required=True, help="Path to the denoiser .pth model.")
    p_ecg.add_argument("--classifier_path", type=str, required=True, help="Path to the classifier .pth model.")
    p_ecg.add_argument("--noise_dir", type=str, required=True, help="Path to the 'mit-bih-noise-stress-test-database-1.0.0' directory.")
    p_ecg.add_argument("--output_prefix", type=str, required=True, help="Prefix for saving output confusion matrices.")
    p_ecg.add_argument("--record_name", type=str, default="201", help="MIT-BIH record name to use for validation.")
    p_ecg.add_argument("--snr_db", type=int, default=0, help="SNR level for the noisy signal.")

    # --- EEG Subparsers ---
    p_eeg_st = subparsers.add_parser("eeg_spatiotemporal", help="Generate topomap comparison video for EEG.")
    p_eeg_st.add_argument("--baseline_model_path", type=str, required=True)
    p_eeg_st.add_argument("--spatial_model_path", type=str, required=True)
    p_eeg_st.add_argument("--test_file_path", type=str, required=True, help="Path to an .edf file for validation.")
    p_eeg_st.add_argument("--output_path", type=str, required=True, help="Path to save the output video.")

    p_eeg_freq = subparsers.add_parser("eeg_frequency", help="Generate PSD comparison plot for EEG.")
    p_eeg_freq.add_argument("--baseline_model_path", type=str, required=True)
    p_eeg_freq.add_argument("--frequency_model_path", type=str, required=True)
    p_eeg_freq.add_argument("--data_dir", type=str, required=True, help="Path to the EEG data directory.")
    p_eeg_freq.add_argument("--output_path", type=str, required=True, help="Path to save the output plot.")

    p_eeg_umap = subparsers.add_parser("eeg_unsupervised", help="Generate UMAP embedding plot from self-supervised model.")
    p_eeg_umap.add_argument("--model_path", type=str, required=True, help="Path to the self-supervised .pth model.")
    p_eeg_umap.add_argument("--data_dir", type=str, required=True, help="Path to the EEG data directory for creating test set.")
    p_eeg_umap.add_argument("--output_path", type=str, required=True, help="Path to save the output plot.")

    args = parser.parse_args()

    # Route to the correct validation function
    if args.validation_task == "ecg_downstream":
        validate_ecg_downstream(args)
    elif args.validation_task.startswith("eeg_"):
        validate_eeg(args)