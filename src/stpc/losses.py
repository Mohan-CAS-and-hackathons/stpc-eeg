# src/stpc/losses.py
import torch
import torch.nn as nn

# --- ECG Losses ---

class GradientLoss(nn.Module):
    """
    Computes the L1 loss between the gradients (forward-difference) of the
    prediction and the target for 1D signals.
    Corresponds to the Temporal-Gradient Consistency term.
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction, target):
        pred_grad = torch.diff(prediction, dim=-1)
        target_grad = torch.diff(target, dim=-1)
        return self.loss(pred_grad, target_grad)

class FFTLoss(nn.Module):
    """
    Computes the L1 loss between the magnitudes of the FFT of the prediction
    and target for 1D signals.
    Corresponds to the Spectral-Magnitude Consistency term.
    """
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction, target):
        pred_fft_mag = torch.abs(torch.fft.fft(prediction, dim=-1))
        target_fft_mag = torch.abs(torch.fft.fft(target, dim=-1))
        return self.loss(pred_fft_mag, target_fft_mag)

# --- EEG Losses ---

class LaplacianLoss(nn.Module):
    """
    Computes the L1 loss between the spatial Laplacians of multi-channel signals.
    Corresponds to the Spatial-Laplacian term for EEG.
    """
    def __init__(self, adj_list):
        super().__init__()
        self.adj_list = adj_list
        self.loss = nn.L1Loss()

    def _calculate_laplacian(self, x):
        L = torch.zeros_like(x)
        for i, neighbors in enumerate(self.adj_list):
            if len(neighbors) > 0:
                L[:, i, :] = x[:, i, :] - torch.mean(x[:, neighbors, :], dim=1)
        return L

    def forward(self, pred, target):
        return self.loss(self._calculate_laplacian(pred), self._calculate_laplacian(target))

class BandMaskedFFTLoss(nn.Module):
    """
    Computes FFT loss but only within a specific frequency band (e.g., Alpha band).
    """
    def __init__(self, fs, f_low, f_high):
        super().__init__()
        self.fs, self.f_low, self.f_high = fs, f_low, f_high
        self.loss = nn.L1Loss()
        self.mask = None

    def forward(self, pred, target):
        if self.mask is None:
            T = pred.shape[-1]
            freqs = torch.fft.rfftfreq(T, 1.0 / self.fs).to(pred.device)
            self.mask = ((freqs >= self.f_low) & (freqs <= self.f_high)).float()[None, None, :]
        
        mag_p = torch.abs(torch.fft.rfft(pred, dim=-1))
        mag_c = torch.abs(torch.fft.rfft(target, dim=-1))
        return self.loss(mag_p * self.mask, mag_c * self.mask)

# Note: The original eeg_train.py had a TemporalGradientLoss which is functionally
# identical to GradientLoss. We will use the shared GradientLoss.