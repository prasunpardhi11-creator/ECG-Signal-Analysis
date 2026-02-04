import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd

# ---------- Load ECG Data ----------
def load_ecg(filename):
    data = pd.read_csv(filename)
    return data['ecg'].values

# ---------- Bandpass Filter ----------
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=250, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

# ---------- R Peak Detection ----------
def detect_r_peaks(signal, fs=250):
    peaks, _ = find_peaks(signal, distance=fs*0.6, height=np.mean(signal))
    return peaks

# ---------- BPM Calculation ----------
def calculate_bpm(peaks, fs=250):
    rr_intervals = np.diff(peaks) / fs
    bpm = 60 / np.mean(rr_intervals)
    return round(bpm, 2)

# ---------- Visualization ----------
def plot_ecg(raw, filtered, peaks):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title("Raw ECG Signal")
    plt.plot(raw)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.title("Filtered ECG with R Peaks")
    plt.plot(filtered)
    plt.plot(peaks, filtered[peaks], "ro")
    plt.grid()

    plt.tight_layout()
    plt.show()

# ---------- Main ----------
if __name__ == "__main__":

    fs = 250   # sampling frequency

    ecg = load_ecg("ecg_sample.csv")

    filtered_ecg = bandpass_filter(ecg, fs=fs)

    r_peaks = detect_r_peaks(filtered_ecg, fs=fs)

    bpm = calculate_bpm(r_peaks, fs=fs)

    print("Detected Heart Rate:", bpm, "BPM")

    plot_ecg(ecg, filtered_ecg, r_peaks)
