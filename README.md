# Sony MSZ-2100G Raw Converter

An open-source decoder and batch converter for the **Sony MSZ-2100G Multispectral
Sensing Unit** (agricultural drone camera). It converts the camera's raw output
into standard images (PNG previews + 16-bit TIFFs) and computes NDVI maps.

The camera's RGB raw format is undocumented, proprietary, and (on firmware 2.00)
block-compressed on-sensor. The formats supported here were reverse-engineered
from real captures — see [docs/FORMAT.md](docs/FORMAT.md) for the full
specification and [docs/TUTORIAL.md](docs/TUTORIAL.md) for a hands-on,
step-by-step decoding walkthrough. As far as we know this is the only decoder
for these files outside Sony's own Fast Field Analyzer software.

## Supported inputs

| File | Camera | Firmware | Format |
|---|---|---|---|
| `cam2_*_RED.raw`, `cam2_*_NIR.raw` | dual-bandpass (RED/NIR) | all | 960×540, uncompressed 12-bit |
| `cam1_*_RGB.raw` | RGB (Bayer) | 2.00 | 4544×2584, Sony on-sensor block compression |
| `cam1_*_RGB.tiff` | RGB (Bayer) | 2.20 | 4-page TIFF, one 8-bit JPEG page per Bayer plane |

## Installation (conda)

```bash
conda env create -f environment.yml
conda activate msz2100g
```

## Usage

Copy session folders off the camera's microSD card (they live under `Sony/` on
the card, named `YYYYMMDDhhmmss`), then:

```bash
python msz_convert.py <raw_root> <output_root>
# e.g.
python msz_convert.py ./raw ./converted
```

The converter walks every session folder under `<raw_root>` and writes, per session:

```
<output_root>/<session>/
  png/    viewable previews (auto-stretched; RGB demosaiced, white-balanced, denoised)
  tif/    16-bit TIFFs with sensor values (RGB = undemosaiced Bayer mosaic on fw2.00;
          original 4-page TIFF copied through on fw2.20)
  ndvi/   NDVI = (NIR-RED)/(NIR+RED) after black-level subtraction, for every
          RED/NIR pair with usable signal in both bands.
          Float32 TIFF (values) + colormapped PNG (red=low → green=high)
```

Already-converted files are skipped, so re-running after adding new sessions is cheap.

## Conversion process (what the tool actually does)

1. **RED/NIR**: memory-map the headerless little-endian `uint16` stream, reshape to
   960×540. Values are 12-bit sensor DNs stored `<<4`; black level is 239 DN
   (matches `rgb/dbp_imager_black_level_value` in the camera's `sens_*.csv` logs).
2. **RGB, firmware 2.00**: decode the 10-byte compression blocks
   (`pixel = (base << 4) + (nibble << (4 + shift))`), de-interleave the
   colour-grouped blocks back into a 4544×2584 RGGB mosaic, then demosaic at
   half resolution for the preview.
3. **RGB, firmware 2.20**: read all four TIFF pages (Gr, R, B, Gb planes),
   reconstruct colour as `G=(p0+p3)/2, R=p1, B=p2` (black level 15).
4. **Previews**: percentile stretch with hot-pixel-robust statistics, bounded
   gray-world white balance, and light luma/chroma denoising for dark frames.
5. **NDVI**: 3×3 box denoise per band, black-level subtraction, per-pixel
   `(NIR-RED)/(NIR+RED)`; pixels with too little combined signal are masked gray.

**Calibration caveat:** NDVI values are *uncalibrated* unless the GNSS sensor
unit logs sunlight-sensor data (`light_sensor_red` / `light_sensor_ir` in
`sprit_*.csv`). Without it, absolute values are shifted; relative comparisons
within a frame remain valid.

## Camera operation notes

- One press of the shutter button starts interval capture (~1 frame/s on the
  prototype, `ShutterInterval` on retail firmware); a second press stops it.
  Each run becomes one session folder.
- The camera clock resets to 2017-07-01 00:00:00 at power-on when there is no
  GNSS fix, so folder timestamps are relative to boot, and session names can
  collide across days — keep captures from different days in separate folders.
- Fixed exposure (RGB 369 µs, RED/NIR 500 µs) — the camera is designed for
  daylight; indoor and dusk scenes record close to black level.
- microSD: FAT32, Class 10; the official spec is SDHC (≤32 GB).

## License

MIT — see [LICENSE](LICENSE). Not affiliated with or endorsed by Sony.
Format details were reverse-engineered from the author's own camera for
interoperability.
