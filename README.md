# Noninvasive fECG extraction 

This repo contains a Python script implementing an end-to-end pipeline for
noninvasive fetal ECG extraction from multichannel maternal abdominal recordings
(PhysioNet Challenge 2013 dataset).

See: PhysioNet/Computing in Cardiology Challenge 2013. :contentReference[oaicite:1]{index=1}

Files:
- fetal_ecg_extraction.py  : main script (usage examples inside)
- requirements.txt         : Python dependencies

Quick start:
1. Create virtualenv and `pip install -r requirements.txt`.
2. Download PhysioNet challenge dataset (set-a or set-b) from PhysioNet (zip or tar) and extract.
3. Run:
   `python fetal_ecg_extraction.py --record path/to/recordname --ann path/to/recordname.fqrs`
   - If you don't have annotation file, omit `--ann` (evaluation will be skipped).

The script demonstrates:
- Preprocessing (HP, notch, optional bandpass)
- Maternal cancellation (template subtraction, LMS adaptive filter)
- ICA-based separation + wavelet denoising
- QRS detection (simple filtered-peaks approach)
- Plots and optional evaluation vs. reference annotations

License: MIT
