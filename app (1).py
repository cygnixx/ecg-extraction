import os, tarfile, glob, requests
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pywt
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from sklearn.decomposition import FastICA
import streamlit as st

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="Noninvasive Fetal ECG", layout="wide")
st.title("🩺 Noninvasive Fetal ECG Extraction (PhysioNet Challenge 2013)")
st.markdown("""
This app downloads abdominal ECG recordings from PhysioNet (Challenge 2013) and demonstrates:
1. Signal preprocessing (band-pass + notch)  
2. Maternal ECG suppression via ICA  
3. Wavelet denoising  
4. Fetal R-peak detection and FHR computation
""")

# ---------------- Helper functions ----------------
def bandpass_filter(sig, fs, low=0.5, high=100.0):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def notch_filter(sig, fs, freq=50.0, Q=30.0):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, sig)

def wavelet_denoise(sig, wavelet='db4', level=3):
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1]))/0.6745
    uth = sigma*np.sqrt(2*np.log(len(sig)))
    coeffs[1:] = [pywt.threshold(c, value=uth, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def download_and_extract(set_name, url):
    base = "physionet_data"
    os.makedirs(base, exist_ok=True)
    archive = f"{set_name}.tar.gz"
    dest = os.path.join(base, set_name)
    if not os.path.exists(archive):
        with st.spinner(f"📦 Downloading {set_name} (~500 MB)... please wait"):
            r = requests.get(url, stream=True)
            with open(archive, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
    if not os.path.exists(dest):
        with st.spinner("📂 Extracting dataset..."):
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path=dest)
    return dest

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls")
dataset = st.sidebar.selectbox("Dataset", ["set-a", "set-b"])
sets_info = {
    "set-a": "https://physionet.org/files/challenge-2013/1.0.0/set-a.tar.gz",
    "set-b": "https://physionet.org/files/challenge-2013/1.0.0/set-b.tar.gz",
}

lowcut = st.sidebar.slider("High-pass (Hz)", 0.1, 5.0, 0.5)
highcut = st.sidebar.slider("Low-pass (Hz)", 40.0, 150.0, 100.0)
notch = st.sidebar.selectbox("Notch frequency", [50.0, 60.0])
wavelet = st.sidebar.text_input("Wavelet", "db4")
wave_level = st.sidebar.slider("Wavelet level", 1, 6, 3)

if st.sidebar.button("Run Analysis"):
    data_folder = download_and_extract(dataset, sets_info[dataset])
    hea_files = glob.glob(os.path.join(data_folder, "**/*.hea"), recursive=True)
    if not hea_files:
        st.error("❌ No .hea files found after download.")
        st.stop()

    record_path = os.path.splitext(hea_files[0])[0]
    record = wfdb.rdrecord(record_path)
    samples, _ = wfdb.rdsamp(record_path)
    fs = record.fs
    st.success(f"✅ Loaded record {os.path.basename(record_path)} with {samples.shape[1]} channels @ {fs} Hz")

    # Limit plotting to 10 seconds
    samples_to_plot = int(min(10 * fs, len(samples)))

    # --- Raw ECG preview ---
    st.subheader("📈 Raw Abdominal ECG (first 10 seconds)")
    fig, axs = plt.subplots(samples.shape[1], 1, figsize=(10, 2*samples.shape[1]))
    for ch in range(samples.shape[1]):
        axs[ch].plot(samples[:samples_to_plot, ch])
        axs[ch].set_ylabel(f"Ch {ch}")
    st.pyplot(fig)

    # --- Filtering ---
    st.subheader("⚙️ Filtering (Band-pass + Notch)")
    filtered = np.zeros_like(samples)
    for ch in range(samples.shape[1]):
        temp = notch_filter(samples[:, ch], fs, freq=notch)
        filtered[:, ch] = bandpass_filter(temp, fs, lowcut, highcut)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(filtered[:samples_to_plot,0])
    ax.set_title("Filtered Abdominal ECG (Channel 0, first 10s)")
    st.pyplot(fig)

    # --- ICA maternal suppression ---
    st.subheader("🧠 Independent Component Analysis (ICA)")
    df = pd.DataFrame(filtered)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(limit_direction='both').fillna(0)
    signals_scaled = (df - df.mean()) / df.std()
    signals_scaled = signals_scaled.fillna(0).values

        # Limit data for ICA to first few seconds to reduce memory use
    max_samples_for_ica = int(min(5 * fs, len(signals_scaled)))   # 5 seconds or less
    subset = signals_scaled[:max_samples_for_ica, :]

    ica = FastICA(
        n_components=subset.shape[1],
        random_state=42,
        max_iter=1000,
        tol=1e-5
    )

    comps = ica.fit_transform(subset)


    fig, axs = plt.subplots(min(6, comps.shape[1]), 1, figsize=(10,2*min(6, comps.shape[1])))
    for i in range(min(6, comps.shape[1])):
        axs[i].plot(comps[:samples_to_plot, i])
        axs[i].set_ylabel(f"IC{i+1}")
    st.pyplot(fig)

   # Use same time segment length for maternal correlation
    maternal = filtered[:comps.shape[0], 0]
    corrs = np.corrcoef(comps.T, maternal)[-1, :-1]
    maternal_idx = np.where(np.abs(corrs)>0.5)[0]
    comps_clean = comps.copy()
    comps_clean[:,maternal_idx]=0
    fetal = (ica.mixing_ @ comps_clean.T).T

    st.subheader("💓 Reconstructed Fetal ECG (first 10s)")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(fetal[:samples_to_plot,0])
    st.pyplot(fig)

    # --- Wavelet denoising + R-peak detection ---
    st.subheader("📊 Fetal R-peak Detection & Heart Rate")
    fetal_ic = comps[:,-1]
    fetal_denoised = wavelet_denoise(fetal_ic, wavelet, wave_level)
    peaks,_ = find_peaks(fetal_denoised, distance=fs*0.3, height=np.std(fetal_denoised))
    rr = np.diff(peaks)/fs

    if len(rr)>0:
        fhr = 60/rr
        fig,ax = plt.subplots(figsize=(10,3))
        ax.plot(fetal_denoised[:samples_to_plot])
        ax.plot(peaks[peaks<samples_to_plot],fetal_denoised[peaks[peaks<samples_to_plot]],'r.')
        ax.set_title("Detected Fetal R-peaks (first 10s)")
        st.pyplot(fig)

        st.line_chart(pd.DataFrame({"FHR (BPM)":fhr}))
        st.write(f"**Mean FHR:** {np.mean(fhr):.1f} BPM ± {np.std(fhr):.1f}")
    else:
        st.warning("⚠️ No R-peaks detected in this segment.")

    st.success("✅ Analysis complete!")

st.markdown("---")
st.caption("BSS Group 2 Final Project — Demo Phase. Based on PhysioNet 2013 Noninvasive Fetal ECG Challenge.")
