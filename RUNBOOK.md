# MSZ-2100G processing runbook (offline)

Everything runs locally with Python + a few libraries. No internet needed.

## One-time setup (do once)

Install Miniconda/Anaconda, then in an Anaconda Prompt:

    conda env create -f environment.yml
    conda activate msz2100g

(Equivalent: `conda create -n msz2100g python=3.11 numpy pillow opencv -y`)

## Everyday use

Open Anaconda Prompt and activate the env each session:

    conda activate msz2100g
    cd <folder containing the scripts>

### 1. Convert a card / session folder to viewable images + NDVI
Point it at a folder that contains session subfolders (each full of cam1/cam2 files):

    python msz_convert.py  <raw_folder>  <output_folder>

Produces, per session:  png/ (viewable)  tif/ (16-bit data)  ndvi/ (maps).
Already-done files are skipped, so re-running after adding sessions is cheap.

### 2. NDVI-over-RGB overlays with legend (needs opencv)
Run on ONE session folder at a time (the folder holding cam1_*/cam2_* files):

    python align_ndvi.py  <session_folder>  <overlay_output_folder>  --calib msz_cam2_to_cam1.json

- `--calib msz_cam2_to_cam1.json` reuses the saved camera-to-camera alignment
  (rotation/scale). Per-frame parallax is corrected automatically.
- Omit `--calib` to estimate a fresh alignment from that session (best for
  close-range / changed setups).
- Output PNGs have the red->yellow->green NDVI legend baked in.

## Gotchas
- Dark/dusk scenes: RGB (8-bit JPEG on fw2.20) looks poor - shoot in daylight.
- 0-byte files at the end of a run = SD card write failure; those frames are lost.
  Reformat the card (FAT32) if you see empty sens_/sprit_ logs or a 0-byte tail.
- NDVI is uncalibrated unless the sensor unit logs light data (sprit_*.csv not "error").
