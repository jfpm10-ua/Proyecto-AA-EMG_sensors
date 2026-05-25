from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, cross_validate
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import entropy as scipy_entropy
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from scipy import signal
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import time


class Model(nn.Module):
    def __init__(
        self,
        n_channels: int,
        hidden_size: int,
        nlayers: int,
        dropout: float
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=nlayers,
            batch_first=True,
            dropout=dropout if nlayers > 1 else 0.0,
            dtype=torch.bfloat16
        )
        self.norm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 4, dtype=torch.bfloat16)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        return self.head(self.drop(out))


class LSTMClassifier(BaseEstimator):
    def __init__(
        self,
        n_channels: int = 64,
        hidden_size: int = 100,
        nlayers: int = 3,
        dropout: float = 0.3,
        epochs: int = 150
    ):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.epochs = epochs
    
    def fit(self, X_train, y_train):
        self.model = Model(self.n_channels, self.hidden_size, self.hidden_size, self.dropout).to("cuda")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        train_dataloader = DataLoader(TensorDataset(X_train, y_train),
                                        batch_size=32, shuffle=True, drop_last=True)

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            train_loss = torch.tensor(0., device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for i, (X, y) in enumerate(train_dataloader):
                    X, y = X.to("cuda"), y.to("cuda")
                    y_pred = self.model(X.to(torch.bfloat16))
                    loss: torch.Tensor = loss_fn(y_pred, y)
                    train_loss += loss
            train_loss /= (i + 1)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            train_loss.backward()
            optimizer.step()
            scheduler.step()

    @torch.no_grad
    def predict(self, X):
        self.model.eval()
        X = X.to("cuda")
        ypred: torch.Tensor = self.model(X.to(torch.bfloat16))
        return torch.argmax(torch.softmax(ypred.cpu().to(torch.float16), dim=-1), dim=1)

    @torch.no_grad
    def score(self, X, y):
        # accuracy = Accuracy(task="multiclass", num_classes=4).to("cuda")
        self.model.eval()
        X = X.to("cuda")
        ypred: torch.Tensor = self.model(X.to(torch.bfloat16))
        return accuracy_score(ypred.cpu().to(torch.float16), y)


SAMPLING_RATE = 200
GESTURE_NAMES = {0: "Rock", 1: "Scissors", 2: "Paper", 3: "OK", 4: "Rest"}
BAND_DEFS = {
    "Very Low (0-20Hz)":    (0,   20),
    "Low (20-80Hz)":        (20,  80),
    "Mid (80-200Hz)":       (80,  200),
    "High (200-500Hz)":     (200, 500),
}


def compute_psd(signal_window: np.ndarray, fs: int = SAMPLING_RATE):
    """
    Compute the Power Spectral Density using Welch's method.
    Returns frequencies (Hz) and power values.
    Welch's method reduces noise by averaging overlapping FFT segments.
    """
    nperseg = min(len(signal_window), 256)
    freqs, psd = signal.welch(signal_window, fs=fs, nperseg=nperseg)
    return freqs, psd


def mean_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Mean Frequency (MNF) — centroid of the power spectrum.
    MNF = Σ(f_i * P_i) / Σ(P_i)

    Interpretation: Higher MNF → more high-frequency content.
    In EMG, MNF decreases with muscle fatigue.
    """
    total_power = np.sum(psd)
    if total_power == 0:
        return 0.0
    return float(np.sum(freqs * psd) / total_power)


def median_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Median Frequency (MDF) — frequency splitting cumulative power in half.
    Σ(P_i, i<MDF) = Σ(P_i, i>MDF) = Total_Power / 2

    Interpretation: More robust to noise than MNF.
    Also decreases with muscle fatigue.
    """
    cumulative_power = np.cumsum(psd)
    half_power = cumulative_power[-1] / 2.0
    idx = np.searchsorted(cumulative_power, half_power)
    idx = np.clip(idx, 0, len(freqs) - 1)
    return float(freqs[idx])


def peak_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Peak Frequency — frequency at which PSD reaches its maximum.

    Interpretation: Identifies the dominant oscillation frequency.
    """
    return float(freqs[np.argmax(psd) % len(freqs)])


def total_power(psd: np.ndarray) -> float:
    """
    Total Power — sum of all power spectral density values.
    Equivalent to signal variance (Parseval's theorem).

    Interpretation: Reflects overall muscle activation level.
    """
    return float(np.sum(psd))


def mean_power(psd: np.ndarray) -> float:
    """
    Mean Power — average power across all frequency bins.

    Interpretation: Normalized version of total power.
    """
    return float(np.mean(psd))


def spectral_moments(freqs: np.ndarray, psd: np.ndarray) -> dict:
    """
    Spectral Moments (M0 – M3) — statistical moments of the PSD.
    M_k = Σ(f_i^k * P_i)

    M0 = Total Power (0th moment)
    M1 = 1st moment (related to MNF)
    M2 = 2nd moment (spread around zero)
    M3 = 3rd moment (spectral skewness)

    Interpretation: Fully characterize spectral shape; used to derive MNF & MDF.
    """
    moments = {}
    for k in range(4):
        moments[f"M{k}"] = float(np.sum((freqs ** k) * psd))
    return moments


def frequency_ratio(freqs: np.ndarray, psd: np.ndarray,
                    low_band=(20, 80), high_band=(80, 500)) -> float:
    """
    Frequency Ratio — ratio of power in a low band vs. a high band.
    FR = Power(low_band) / Power(high_band)

    Default: Low = 20-80 Hz, High = 80-500 Hz

    Interpretation: FR > 1 → energy concentrated at lower frequencies.
    Useful for distinguishing gesture types and fatigue states.
    """
    def band_power_calc(f, p, fmin, fmax):
        mask = (f >= fmin) & (f <= fmax)
        return np.sum(p[:, mask])

    low_p = band_power_calc(freqs, psd, *low_band)
    high_p = band_power_calc(freqs, psd, *high_band)
    if high_p == 0:
        return 0.0
    return float(low_p / high_p)


def spectral_entropy(psd: np.ndarray) -> float:
    """
    Spectral Entropy — Shannon entropy of the normalized PSD.
    SE = -Σ(p_i * log(p_i))  where p_i = P_i / Σ(P_i)

    Interpretation:
      - Low entropy → power concentrated in few frequencies (periodic signal).
      - High entropy → power spread across many frequencies (random/complex signal).
    """
    psd_norm = psd / (np.sum(psd) + 1e-12)
    return np.mean(scipy_entropy(psd_norm + 1e-12))


def extract_frequency_features(signal_window: np.ndarray,
                                fs: int = SAMPLING_RATE) -> dict:
    """
    Extract ALL frequency features from a single-channel signal window.
    Returns a flat dictionary of feature_name → value.
    """
    freqs, psd = compute_psd(signal_window, fs)

    features = []

    features.append(mean_frequency(freqs, psd))
    features.append(median_frequency(freqs, psd))
    features.append(peak_frequency(freqs, psd))
    features.append(total_power(psd))
    features.append(mean_power(psd))
    features.append(frequency_ratio(freqs, psd))
    features.append(spectral_entropy(psd))

    for val in spectral_moments(freqs, psd).values():
        features.append(val)

    return np.array(features)


def load_dataset(n_channels: int, block_size: int):
    all_windows, all_labels = [], []
    n_classes = 4

    for gesture_id in range(n_classes):
        df = pd.read_csv(f"data/{gesture_id}.csv", header=None)
        # shape: (T, 64)  — drop last column (label), keep all sensor channels
        signal = df.iloc[:, :n_channels].values.astype(np.float32)  # (T, 64)

        n_windows = len(signal) // block_size
        signal = signal[:n_windows * block_size]          # trim remainder
        wins = signal.reshape(n_windows, block_size, n_channels)  # (W, 15, 64)

        all_windows.append(wins)
        all_labels.append(np.full(n_windows, gesture_id, dtype=np.int64))

    X = torch.from_numpy(np.concatenate(all_windows))   # (total_W, 15, 64)
    y = torch.from_numpy(np.concatenate(all_labels))    # (total_W,)
    return X, y


class Loader:
    def __init__(self, data, labels, batch_size: int, block_size: int, accum: bool = False, max_accum: int = 0):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.block_size = block_size
        self.accum = accum
        self.max_accum = max_accum
        self.blocks = []
        self.i = 0
        for i in range(np.prod(self.data.shape) // (self.batch_size * self.block_size)):
            bli = i % (self.data.shape[1] // self.block_size)
            bai = i // (self.data.shape[1] // self.block_size)
            v = self.data[bai * self.batch_size:(bai + 1) * self.batch_size, bli * self.block_size:(bli + 1) * self.block_size]
            l = self.labels[bai * self.batch_size:(bai + 1) * self.batch_size]
            self.blocks.append((v, l))
        n = len(self.blocks)
        self.order = np.random.choice(np.arange(n), size=n, replace=False)
        if accum:
            self.idx = 0
            self.cicle = 1
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.accum:
            if self.idx >= len(self.order):
                self.idx = 0
                self.cicle = 1
            while self.idx < self.max_accum * self.cicle:
                idx = self.order[self.idx]
                self.idx += 1
                return self.blocks[idx]
            self.cicle += 1
            raise StopIteration
        else:
            while self.i < len(self.order):
                idx = self.order[self.i]
                self.i += 1
                return self.block_size[idx]
            self.i = 0
            raise StopIteration()


def eval(
    model: Model,
    dataloader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    accuracy: Accuracy,
    epoch: int,
    epochs: int,
    show: bool = True,
    noise: bool = False,
    display: bool = True
):
    model.eval()
    test_loss = torch.tensor(0., device="cuda")
    test_acc = torch.tensor(0., device="cuda")
    with torch.inference_mode():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to("cuda").to(torch.bfloat16), y.to("cuda")
            y_pred = model(X)
            loss: torch.Tensor = loss_fn(y_pred, y)
            acc: torch.Tensor = accuracy(y_pred, y)
            test_loss += loss.detach()
            test_acc += acc.detach()
        test_loss /= (i + 1)
        test_acc /= (i + 1)

    if display:
        if not noise:
            if show:
                print(f"[{epoch:>{len(str(epochs))}}/{epochs}] [Test] Loss: {test_loss:.2f} | Accuracy: {test_acc * 100:.2f}%")
            else:
                print(f"[Test] Loss: {test_loss:.2f} | Accuracy: {test_acc * 100:.2f}%")
        else:
            print(f"[Noise] Loss: {test_loss:.2f} | Accuracy: {test_acc * 100:.2f}%")
    return test_loss, test_acc


def train(
    X_train, y_train,
    X_test, y_test,
    hidden_size: int = 100,
    nlayers: int = 5,
    dropout: float = 0.3,
    epochs: int = 150,
    show: bool = True
):
    if show:
        print(f"Loading dataset")
    np.random.seed(42)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    noise = np.random.uniform(-0.3*torch.std(X_test), 0.3*torch.std(X_test), X_test.shape)
    X_noise = X_test + noise

    model = Model(n_channels=n_channels, hidden_size=hidden_size, nlayers=nlayers, dropout=dropout).to("cuda")
    loss_fn = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=4).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_dataloader = DataLoader(TensorDataset(X_train, y_train),
                                    batch_size=32, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test),
                                    batch_size=32, shuffle=False)
    noise_dataloader = DataLoader(TensorDataset(X_noise, y_test),
                                    batch_size=32, shuffle=False)

    start = time.time()
    for epoch in range(epochs):
        if epoch % 50 == 0:
            tel, tea = eval(model, test_dataloader, loss_fn, accuracy, epoch, epochs, display=show)
            test_loss.append(tel.item())
            test_acc.append(tea.item())

        model.train()
        optimizer.zero_grad()
        tl = torch.tensor(0., device="cuda")
        ta = torch.tensor(0., device="cuda")
        for i, (X, y) in enumerate(train_dataloader):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                X, y = X.to("cuda").to(torch.bfloat16), y.to("cuda")
                y_pred = model(X)
                loss: torch.Tensor = loss_fn(y_pred, y)
                acc: torch.Tensor = accuracy(y_pred, y)
            tl += loss.detach()
            ta += acc.detach()            
            loss.backward()
        tl /= (i + 1)
        ta /= (i + 1)
        train_loss.append(tl.item())
        train_acc.append(ta.item())
        optimizer.step()
        scheduler.step()
        if show:
            print(f"[{epoch:>{len(str(epochs))}}/{epochs}] [Train] Loss: {tl:.2f} | Accuracy: {ta * 100:.2f}%")
    end = time.time()

    if show:
        print()
        print("Finished training.")
        eval(model, test_dataloader, loss_fn, accuracy, -1, -1, False)
        eval(model, noise_dataloader, loss_fn, accuracy, -1, -1, False, noise=True)
        print(f"Training time: {end - start:.2f} s")
    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }, model

def calculate_latency(model, X_test):
    latencies = []
    for _ in range(5):
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                start = time.time()
                X_test = X_test.to("cuda").to(torch.bfloat16)
                y_pred = model(X_test)
                end = time.time()
                latencies.append(end - start)
    elapsed = np.mean(latencies)
    return elapsed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Whether or not to use frequency features (raw vs. features)
    parser.add_argument("-f", "--features", action="store_true")
    args = parser.parse_args()

    n_channels = 64
    block_size = 15
    data, labels = load_dataset(n_channels, block_size)
    if args.features:
        d = []
        for row in data:
            d.append(extract_frequency_features(row.numpy(), fs=200))
        data = torch.from_numpy(np.vstack(d)).unsqueeze(1)
        n_channels = 11
    X_train: torch.Tensor
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.25, random_state=37, stratify=labels
    )
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True) + 1e-8
    X_test = (X_test - mean) / std
    noise = np.random.uniform(-0.3*torch.std(X_test), 0.3*torch.std(X_test), X_test.shape)
    X_noise = X_test + noise

    best_model = None
    best_acc = float("-inf")
    best_params = ()
    info = {}
    for epochs in [150, 300, 500]:
        for nlayers in [1, 2, 3, 5]:
            for hidden_size in [50, 100, 150]:
                for dropout in [0.0, 0.1, 0.3]:
                    print(f"Epochs: {epochs} | # layers: {nlayers} | Hidden Size: {hidden_size} | Dropout: {dropout}")
                    for _ in range(5):
                        X_val_train, X_val, y_val_train, y_val = train_test_split(
                            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                        )
                        X_val_train = (X_val_train - mean) / std
                        X_val = (X_val - mean) / std
                        info, model = train(
                            X_val_train, y_val_train,
                            X_val, y_val,
                            hidden_size=hidden_size,
                            nlayers=nlayers,
                            dropout=dropout,
                            epochs=epochs,
                            show=False
                        )
                        if info["test_acc"][-1] > best_acc:
                            best_model = model
                            best_acc = info["test_acc"][-1]
                            best_params = (epochs, nlayers, hidden_size, dropout)
                            info = info
    print("Best parameters:")
    print(f"Epochs: {best_params[0]} | # layers: {best_params[1]} | Hidden Size: {best_params[2]} | Dropout: {best_params[3]}")

    time_lstm = calculate_latency(model, X_test)
    time_lstm_n = calculate_latency(model, X_noise)
    lat_lstm = (time_lstm + time_lstm_n) / 2
    print(f"Mean latency of the model: {lat_lstm} s")


    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    xtr = np.arange(len(info["train_loss"]))
    xte = np.arange(0, len(info["train_acc"]), 50)
    ax[0].plot(xtr, info["train_loss"], c="royalblue", linewidth=3, label="Train Loss")
    ax[0].plot(xte, info["test_loss"], c="orangered", linewidth=3, label="Test Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss evolution")
    ax[0].grid(True)

    ax[1].plot(xtr, info["train_acc"], c="royalblue", linewidth=3, label="Train Acc")
    ax[1].plot(xte, info["test_acc"], c="orangered", linewidth=3, label="Test Acc")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_title("Accuracy evolution")
    ax[1].grid(True)
    plt.show()