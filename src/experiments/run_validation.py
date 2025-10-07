# src/experiments/run_validation.py
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np
import wfdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# ... other imports ...

# --- SETUP: Add project root to system path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import from our STPC library ---
from src.stpc.model import UNet1D, ECGClassifier
from src.stpc.utils.ecg_utils import TARGET_FS as ECG_FS, BEAT_CLASSES, BEAT_WINDOW_SIZE, get_noise_signals, load_and_resample_signal as load_ecg_signal, working_directory
from src.stpc.utils.eeg_utils import load_eeg_from_edf, create_eeg_segments, TARGET_FS as EEG_FS
# ... other helper/plotting function imports ...

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def validate_ecg_downstream(cfg: DictConfig):
    print("--- Running End-to-End ECG Downstream Validation ---")
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(os.path.dirname(cfg.paths.output_prefix), exist_ok=True)

    denoiser = UNet1D(in_channels=1, out_channels=1).to(DEVICE)
    denoiser.load_state_dict(torch.load(cfg.paths.denoiser_path, map_location=DEVICE)); denoiser.eval()
    classifier = ECGClassifier().to(DEVICE)
    classifier.load_state_dict(torch.load(cfg.paths.classifier_path, map_location=DEVICE)); classifier.eval()
    
    # The rest of the validation logic remains the same, but uses `cfg`
    with working_directory(cfg.paths.data_dir):
        clean_signal = load_ecg_signal(cfg.run_params.record_name, target_fs=ECG_FS)
        # ... and so on ...

def validate_eeg(cfg: DictConfig):
    print(f"--- Running EEG Validation: {cfg.run_params.eeg_experiment_type} ---")
    print(OmegaConf.to_yaml(cfg))
    
    # The logic for spatial, frequency, and self-supervised validation remains,
    # but is now driven by parameters from the `cfg` object.
    # For example:
    if cfg.run_params.eeg_experiment_type == 'spatial':
        clean_data, _, info = load_eeg_from_edf(cfg.paths.test_file_path)
        # ...
        baseline_model.load_state_dict(torch.load(cfg.paths.baseline_model_path, map_location=DEVICE))
        # ... etc ...

# --- Hydra Entry Point ---
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment_cfg = cfg.experiment

    if "ecg_downstream" in experiment_cfg.experiment_name:
        validate_ecg_downstream(experiment_cfg)
    elif "eeg_validation" in experiment_cfg.experiment_name:
        validate_eeg(experiment_cfg)
    else:
        raise ValueError(f"Unknown validation experiment: {experiment_cfg.experiment_name}")

if __name__ == "__main__":
    main()