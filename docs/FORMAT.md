# MSZ-2100G raw formats (reverse-engineered)

All formats are headerless or standard-container; sessions live in folders
named `YYYYMMDDhhmmss` alongside two CSV logs (`sens_*.csv` per-frame camera
state, `sprit_*.csv` GNSS/light-sensor state).

## cam2: RED / NIR (`cam2_*_RED.raw`, `cam2_*_NIR.raw`) — all firmware

- 1,036,800 bytes exactly: 960 × 540 × uint16 little-endian, no header.
- 12-bit sensor DNs stored left-shifted by 4 (low nibble always 0).
- Black level: 239 (12-bit DN scale), i.e. 3824 in stored units.
- Sensor: 1/2.8-type Exmor R, ~518k effective pixels (matches 960×540).

## cam1: RGB, firmware 2.00 (`cam1_*_RGB.raw`)

- 7,338,560 bytes = 733,856 blocks × 10 bytes, no header.
- Image: 4544 × 2584 Bayer RGGB (R at even row/even col), 12-bit DNs, black 239.
- Each 10-byte block encodes 16 same-colour pixels:
  - byte 0: `base` — block minimum >> 4
  - byte 1: `shift` — adaptive range shift (0–2 observed)
  - bytes 2–9: sixteen 4-bit deltas, low nibble first
  - **pixel = (base << 4) + (nibble << (4 + shift))**
- Block layout: a stored row is 568 blocks covering two sensor rows.
  - First 284 blocks → even sensor row `2r`: blocks alternate
    [16×R][16×Gr]; the 16 same-colour pixels sit 2 apart, so an R+Gr block
    pair covers 32 consecutive sensor columns.
  - Last 284 blocks → odd sensor row `2r+1`: alternating [16×Gb][16×B].
- The encoding is lossy: intra-block values quantize to 16·2^shift-count steps.
- Validation: the two green phases of the reconstructed mosaic agree to <0.5%;
  decoded frames cross-correlate with the (uncompressed) cam2 image of the
  same instant; dark-frame reconstruction matches the logged black level.

## cam1: RGB, firmware 2.20 (`cam1_*_RGB.tiff`)

- A standards-compliant multi-page TIFF containing **four pages**, each
  2272 × 1292, 8-bit single-channel, JPEG-compressed (compression tag 7).
- Pages are the Bayer colour planes in order **Gr, R, B, Gb**
  (i.e. the four phases of the fw2.00 4544×2584 mosaic, one file per plane).
- Black level ≈ 15 (the 12-bit black 239 >> 4).
- Colour reconstruction: `R = page1`, `G = (page0 + page3)/2`, `B = page2`.
- Most image viewers display only page 0 (a green plane), which is why these
  files look like dark grayscale images at first glance.
- Compared with fw2.00 raws: same effective colour resolution as a half-res
  demosaic, but bit depth drops 12 → 8 and JPEG artifacts are introduced.
  Files are ~15× smaller.

## Session CSV logs

- `sens_*.csv`: one row per captured frame (ID = Unix seconds): firmware
  version, serial, supply voltage, temperature, black levels, exposure times
  (RGB 369 µs, DBP 500 µs), gains (RGB 12.2 dB, DBP 0 dB), accelerometer.
- `sprit_*.csv`: GNSS/IMU/light-sensor rows; `light_sensor_red/green/blue/ir`
  enable NDVI illumination calibration when the sensor unit has a fix.
  Rows read `error` when the sensor unit is disconnected or has no data.
