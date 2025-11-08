import os
import tarfile
import glob
import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import requests
import wfdb

from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from sklearn.decomposition import FastICA

import streamlit as st

st.set_page_config(page_title="Fetal ECG (PhysioNet 2013) — Interactive", layout="wide")

# ----------------------------
# Utilities & Caching
# ----------------------------
@st.cache_data(show_spinner=False)
def download_and_extract(set_name: str, url: str, dest_folder: str = "physionet_data") -> str:
    """Download and extract the tar.gz (if not already) and return data folder."""
    os.makedirs(dest_folder, exist_ok=True)
    archive_path = f"{set_name}.tar.gz"
    data_folder = os.path.join(dest_folder, set_name)

    if not os.path.exists(archive_path):
        with st.spinner(f"Downloading {archive_path} ..."):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(archive_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    if not os.path.exists(data_folder):
        with st.spinner(f"Extracting {archive_path} ..."):
            os.makedirs(data_folder, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as tar:
                # safe extract
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

                for member in tar.getmembers():
                    member_path = os.path.join(data_folder, member.name)
                    if not is_within_directory(data_folder, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
                tar.extractall(path=data_folder)
    return data_folder

@st.cache_data(show_spinner=False)
def find_records_in_folder(data_folder: str) -> List[str]:
    hea_files = glob.glob(os.path.join(data_folder, "**/*.hea"), recursive=True)
    record_names = [os.path.splitext(os.path.basename(f))[0] for f in hea_files]
    return record_names

@st.cache_data(show_spinner=False)
def load_wfdb_record(record_path: str):
    """Load a wfdb record by path (without .hea extension)."""
    try:
        record = wfdb.rdrecord(record_path)
        samples, fields = wfdb.rdsamp(record_path)
        return record, samples, fields
    except Exception as e:
        st.error(f"Failed to load {record_path}: {e}")
        return None, None, None

# ----------------------------
# Filters & Processing
# ----------------------------
def bandpass_filter(signal: np.ndarray, fs: float, lowcut=0.5, highcut=100.0, order=4):
    b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal: np.ndarray, fs: float, freq=50.0, Q=30.0):
    b, a = iirnotch(freq/(fs/2), Q)
    return filtfilt(b, a, signal)

def wavelet_denoise(signal: np.ndarray, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

# ----------------------------
# Streamlit layout: sidebar
# ----------------------------
st.sidebar.title("Controls")

# Dataset download options
st.sidebar.markdown("### Dataset")
sets_info = {
    "set-a": "https://physionet.org/files/challenge-2013/1.0.0/set-a.tar.gz",
    "set-b": "https://physionet.org/files/challenge-2013/1.0.0/set-b.tar.gz"
}
dataset_choice = st.sidebar.selectbox("Choose dataset to download/explore", list(sets_info.keys()))

if st.sidebar.button("Download & Extract Selected Set"):
    with st.spinner("Downloading and extracting..."):
        folder = download_and_extract(dataset_choice, sets_info[dataset_choice])
        st.success(f"Downloaded & extracted to {folder}")

# Optionally allow local WFDB folder
local_folder = st.sidebar.text_input("Or enter local WFDB folder path (optional):", value="")

data_folder = local_folder.strip() if local_folder.strip() else os.path.join("physionet_data", dataset_choice)
st.sidebar.markdown(f"**Data folder:** `{data_folder}`")

# List records
records = []
if os.path.isdir(data_folder):
    records = find_records_in_folder(data_folder)
else:
    st.sidebar.info("Dataset folder not found yet. Either download or provide a local path.")

record_selected = st.sidebar.selectbox("Select record", options=records if records else ["(none)"])

# Preprocessing parameters
st.sidebar.markdown("### Filtering")
lowcut = st.sidebar.number_input("High-pass (Hz)", value=0.5, min_value=0.01, step=0.1)
highcut = st.sidebar.number_input("Low-pass (Hz)", value=100.0, min_value=1.0, step=1.0)
notch_freq = st.sidebar.selectbox("Notch frequency", options=[50.0, 60.0], index=0)
apply_notch = st.sidebar.checkbox("Apply notch filter", value=True)

# ICA params
st.sidebar.markdown("### ICA")
ica_max_iter = st.sidebar.number_input("ICA max iterations", value=1000, min_value=100, step=100)
ica_tol = st.sidebar.number_input("ICA tol", value=1e-5, format="%.1e")

# Wavelet / peak detection
st.sidebar.markdown("### Wavelet & Peak Detection")
wavelet = st.sidebar.text_input("Wavelet", value="db4")
wavelet_level = st.sidebar.number_input("Wavelet level", value=3, min_value=1, step=1)
peak_min_distance_s = st.sidebar.number_input("Min peak distance (s)", value=0.3, min_value=0.1, step=0.05)
peak_height_std = st.sidebar.number_input("Peak height threshold (multiple of signal std)", value=1.0, min_value=0.1, step=0.1)

# ----------------------------
# Main content
# ----------------------------
st.header("🩺 Noninvasive Fetal ECG — Interactive (PhysioNet 2013)")
st.markdown("""
This demo:
- Loads multichannel abdominal ECG records.
- Applies notch + bandpass filtering.
- Uses ICA to separate sources and suppress maternal ECG.
- Lets you select a fetal IC, apply wavelet denoising, detect fetal R-peaks and view FHR.
""")

if record_selected == "(none)":
    st.info("No record selected. Download a dataset or provide a local path to WFDB files in the sidebar.")
    st.stop()

# Load selected record
record_path = None
for hea in glob.glob(os.path.join(data_folder, "**/*.hea"), recursive=True):
    name = os.path.splitext(os.path.basename(hea))[0]
    if name == record_selected:
        record_path = os.path.splitext(hea)[0]
        break

if record_path is None:
    st.error("Selected record file not found on disk.")
    st.stop()

record, samples, fields = load_wfdb_record(record_path)
if samples is None:
    st.error("Failed to load record.")
    st.stop()

signals = np.asarray(samples, dtype=float)
fs = record.fs if hasattr(record, "fs") else float(fields.get("fs", 1000.0))
record_name = os.path.basename(record_path)

st.subheader(f"Loaded record: {record_name}")
st.write(f"Signal shape: {signals.shape} (samples × channels), Sampling frequency: {fs} Hz")

# Quick channel selection and preview
cols = st.columns([1, 2])
with cols[0]:
    ch_to_plot = st.number_input("Channel to preview (index)", min_value=0, max_value=signals.shape[1]-1, value=0, step=1)
    preview_seconds = st.number_input("Preview seconds", value=10, min_value=1, max_value=60, step=1)

with cols[1]:
    if st.button("Show raw preview"):
        samples_to_plot = int(preview_seconds * fs)
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(signals[:samples_to_plot, ch_to_plot])
        ax.set_title(f"{record_name} — Channel {ch_to_plot} (first {preview_seconds} s)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        st.pyplot(fig)

# ----------------------------
# Filtering
# ----------------------------
st.markdown("## 1) Filtering")
with st.spinner("Applying filters..."):
    filtered_signals = np.zeros_like(signals)
    for ch in range(signals.shape[1]):
        sig = signals[:, ch]
        if apply_notch:
            try:
                sig = notch_filter(sig, fs, freq=notch_freq)
            except Exception:
                # if notch implementation parameters differ, fallback to no notch
                pass
        sig = bandpass_filter(sig, fs, lowcut=lowcut, highcut=highcut)
        filtered_signals[:, ch] = sig

st.success("Filtering complete.")

# Show raw vs filtered for preview range
samples_to_plot = int(preview_seconds * fs)
fig, axs = plt.subplots(2, 1, figsize=(12, 4))
axs[0].plot(signals[:samples_to_plot, ch_to_plot])
axs[0].set_title("Raw")
axs[0].grid(True)
axs[1].plot(filtered_signals[:samples_to_plot, ch_to_plot])
axs[1].set_title("Filtered (bandpass + optional notch)")
axs[1].grid(True)
st.pyplot(fig)

# ----------------------------
# ICA: maternal suppression
# ----------------------------
st.markdown("## 2) ICA (maternal suppression)")
with st.spinner("Running ICA... (this may take a few seconds)"):
    # handle constant columns
    df = pd.DataFrame(filtered_signals).interpolate().fillna(0).values
    nonzero_var_idx = np.std(df, axis=0) > 0
    signals_clean = df[:, nonzero_var_idx]

    # scale
    signals_scaled = (signals_clean - np.mean(signals_clean, axis=0)) / (np.std(signals_clean, axis=0) + 1e-9)

    n_components = signals_scaled.shape[1]
    ica = FastICA(n_components=n_components, random_state=42, max_iter=int(ica_max_iter), tol=float(ica_tol))
    components = ica.fit_transform(signals_scaled)  # shape: (n_samples, n_components)
    mixing = ica.mixing_  # shape: (n_features, n_components)

st.success("ICA complete.")

# Show first few components preview
n_plot_ic = min(6, components.shape[1])
fig, axs = plt.subplots(n_plot_ic, 1, figsize=(12, 2.0*n_plot_ic), sharex=True)
for i in range(n_plot_ic):
    axs[i].plot(components[:samples_to_plot, i])
    axs[i].set_ylabel(f"IC{i+1}")
axs[-1].set_xlabel("Samples")
st.pyplot(fig)

# Attempt to auto-identify maternal ICs using correlation with first channel (assumed maternal chest)
maternal_channel = filtered_signals[:, 0]  # approximate
corrs = np.array([np.corrcoef(components[:, i], maternal_channel)[0,1] for i in range(components.shape[1])])
abs_corrs = np.abs(corrs)
auto_maternal_idx = list(np.where(abs_corrs > 0.5)[0])  # threshold can be tuned
st.write("IC correlations with channel 0 (approx maternal):")
st.write(pd.DataFrame({"IC": np.arange(len(corrs)), "corr": corrs}))

st.write("Auto-identified maternal ICs (|corr| > 0.5):", auto_maternal_idx)

# Allow user to override maternal IC removal
st.write("Select maternal ICs to zero out (suppress):")
maternal_ics_user = st.multiselect("Maternal IC indices to remove", options=list(range(components.shape[1])), default=auto_maternal_idx)

# Zero out chosen components and reconstruct signals
components_clean = components.copy()
components_clean[:, maternal_ics_user] = 0
reconstructed = mixing.dot(components_clean.T).T  # shape -> (n_samples, n_features)
# Map back to original channel indices (nonzero_var_idx)
reconstructed_full = np.zeros_like(df)
reconstructed_full[:, nonzero_var_idx] = reconstructed

# Show reconstructed fetal candidate channel (first feature)
st.markdown("### Reconstructed (maternal suppressed) signals — channel preview")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(reconstructed_full[:samples_to_plot, 0])
ax.set_title("Reconstructed signal (channel 0 preview)")
ax.set_xlabel("Samples")
ax.grid(True)
st.pyplot(fig)

# ----------------------------
# 3) Fetal IC selection & denoising
# ----------------------------
st.markdown("## 3) Fetal IC selection & wavelet denoising")
st.write("Choose an IC that appears fetal (displayed above). Typically an IC with QRS-like spikes is fetal.")
fetal_ic_index = st.number_input("Fetal IC index to use for denoising", min_value=0, max_value=components.shape[1]-1, value=max(0, components.shape[1]-1))
fetal_ic = components[:, fetal_ic_index]

# Wavelet denoise
fetal_denoised = wavelet_denoise(fetal_ic, wavelet=wavelet, level=wavelet_level)
# Trim/pad to original length if needed
if fetal_denoised.shape[0] > fetal_ic.shape[0]:
    fetal_denoised = fetal_denoised[:fetal_ic.shape[0]]
elif fetal_denoised.shape[0] < fetal_ic.shape[0]:
    fetal_denoised = np.pad(fetal_denoised, (0, fetal_ic.shape[0]-fetal_denoised.shape[0]), 'constant')

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(fetal_ic[:samples_to_plot], label="Original IC")
ax.plot(fetal_denoised[:samples_to_plot], label="Denoised (wavelet)", alpha=0.8)
ax.legend()
ax.set_title(f"IC {fetal_ic_index} — raw vs denoised (first {preview_seconds}s)")
ax.grid(True)
st.pyplot(fig)

# ----------------------------
# 4) Peak detection & FHR
# ----------------------------
st.markdown("## 4) Fetal R-peak detection & FHR")
min_distance_samples = int(peak_min_distance_s * fs)
height_threshold = peak_height_std * np.std(fetal_denoised)

peaks, properties = find_peaks(fetal_denoised, distance=min_distance_samples, height=height_threshold)

st.write(f"Detected peaks: {len(peaks)}")

# Show detected peaks on denoised signal (preview)
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(fetal_denoised[:samples_to_plot], label="Denoised fetal IC")
peaks_in_view = peaks[peaks < samples_to_plot]
ax.plot(peaks_in_view, fetal_denoised[peaks_in_view], 'r.', label="Detected R-peaks")
ax.legend()
ax.set_title("Detected R-peaks (preview)")
st.pyplot(fig)

# Compute RR intervals and FHR
if len(peaks) >= 2:
    rr_intervals = np.diff(peaks) / fs
    fhr = 60.0 / rr_intervals
    # Show FHR trend
    st.line_chart(pd.DataFrame({"FHR (BPM)": fhr}))
    st.write("FHR statistics:")
    st.write({
        "beats_detected": int(len(peaks)),
        "mean_FHR": float(np.mean(fhr)),
        "std_FHR": float(np.std(fhr)),
        "min_FHR": float(np.min(fhr)),
        "max_FHR": float(np.max(fhr)),
    })
else:
    st.warning("Not enough peaks detected to compute FHR.")

# ----------------------------
# Wrap-up and Exports
# ----------------------------
st.markdown("---")
st.markdown("### Export")
if st.button("Download denoised fetal IC (CSV)"):
    buf = io.BytesIO()
    df_out = pd.DataFrame({"fetal_denoised": fetal_denoised})
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Click to download CSV", data=buf, file_name=f"{record_name}_fetal_ic{fetal_ic_index}_denoised.csv", mime="text/csv")

st.markdown("**Notes:** This is a demonstration pipeline. Results depend heavily on the chosen record, channel geometry, filtering parameters, and ICA performance. Use this as a starting point for research or prototyping.")
