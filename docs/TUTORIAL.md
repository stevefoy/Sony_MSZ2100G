# Decoding MSZ-2100G files by hand — a tutorial

This walks through decoding each format from first principles, the same way it
was originally reverse-engineered. All snippets need only `numpy` and `pillow`
(`conda env create -f ../environment.yml`). Grab any session folder from the
camera's card and follow along.

## 1. The easy one: RED / NIR band files

Every `cam2_*_RED.raw` / `cam2_*_NIR.raw` is exactly **1,036,800 bytes**.
First instinct with an unknown raw file: factor the size.

```
1,036,800 = 2 x 518,400          -> 16-bit pixels
  518,400 = 960 x 540            -> the sensor's ~518k pixel count
```

Look at the first bytes (`xxd file.raw | head`): pairs like `10 0f`, `00 0f`,
`f0 0e` — little-endian uint16 values 3856, 3840, 3824… all multiples of 16.
That means the low 4 bits are always zero: **12-bit data stored << 4**.

```python
import numpy as np
img = np.fromfile('cam2_..._RED.raw', dtype='<u2').reshape(540, 960)
dn  = img / 16.0          # true 12-bit digital numbers
dn -= 239                 # subtract black level (see sens_*.csv)
```

The black level (239) isn't a guess — the camera logs it per frame in
`sens_*.csv` (`dbp_imager_black_level_value`). A dark frame reads ~239±4 DN.

## 2. NDVI from a RED/NIR pair

```python
load = lambda p: np.fromfile(p, dtype='<u2').reshape(540,960)/16.0 - 239
red, nir = load('..._RED.raw'), load('..._NIR.raw')
ndvi = (nir - red) / (nir + red).clip(1e-6)
```

Caveats: denoise first (3x3 mean), mask pixels where `red+nir` is near zero,
and remember this is *uncalibrated* — proper reflectance calibration needs the
sunlight-sensor columns from `sprit_*.csv` (only logged with a GNSS fix).

## 3. The hard one: fw 2.00 `cam1_*_RGB.raw`

File size **7,338,560 = 733,856 x 10** — ten-byte records. Hexdump a dark
frame and the structure jumps out: every 10 bytes start `0e 00`:

```
0e 00 11 11 10 01 11 11 00 11   <- block: base=14, shift=0, 16 nibbles
```

Each 10-byte block encodes **16 pixels of one Bayer colour**:

| bytes | meaning |
|---|---|
| 0 | `base` = block minimum >> 4 |
| 1 | `shift` = quantization step selector (0–2 seen) |
| 2–9 | sixteen 4-bit deltas, low nibble first |

```python
pixel = (base << 4) + (nibble << (4 + shift))     # 12-bit DN
```

Sanity checks that pinned this down:
* dark frames: base=14, nibbles~1, shift=0 -> 224 + 16 = 240 = the logged black level;
* 733,856 blocks x 16 px = 11,741,696 = 4544 x 2584 ("approx. 12 MP" per Sony's spec sheet);
* the two green Bayer phases of the result agree to <0.5%.

The spatial layout: each stored row of 568 blocks covers TWO sensor rows.
Blocks alternate colours along the row — [16xR][16xGr] pairs for the even
sensor row (first 284 blocks), [16xGb][16xB] for the odd row (last 284).
Same-colour pixels sit 2 apart on the sensor, so one R+Gr block pair spans 32
columns. See `decode_rgb()` in `msz_convert.py` for the exact numpy reshape.

```python
b = np.fromfile(f, dtype=np.uint8).reshape(-1, 10)
base, shift = b[:,0].astype(int), b[:,1].astype(int)
nib = np.empty((len(b),16), int)
nib[:,0::2] = b[:,2:] & 0xF
nib[:,1::2] = b[:,2:] >> 4
px = (base[:,None] << 4) + (nib << (4 + shift[:,None]))   # (blocks, 16)
```

Then reshape into the mosaic (see `decode_rgb`) and demosaic RGGB as usual.
Note the format is lossy: within a block, values quantize to steps of
`16 << shift` — that's Sony's on-sensor compression, not a decoder artifact.

## 4. fw 2.20 `cam1_*_RGB.tiff` — don't trust your image viewer

After the 2.20 firmware update the RGB camera writes TIFF. Open one in a
viewer and you see a dark **grayscale** image — but the file is 4x bigger than
its first image needs. The catch: it's a **multi-page TIFF with four pages**,
and viewers show only page 0.

```python
from PIL import Image
im = Image.open('cam1_..._RGB.tiff')
planes = []
for k in range(4):
    im.seek(k)
    planes.append(np.asarray(im).astype(float))
```

The pages are the four Bayer planes, order **Gr, R, B, Gb** (identified by
correlation: pages 0 and 3 match each other almost perfectly = the two greens).
Each is 2272x1292, 8-bit JPEG-compressed, black level ~15 (= 239 >> 4).

```python
G = (planes[0] + planes[3]) / 2
rgb = np.stack([planes[1]-15, G-15, planes[2]-15], -1).clip(0)
```

White-balance, gamma, done — full colour at the same effective resolution as
the old format's half-res demosaic (but 8-bit + JPEG instead of 12-bit).

## 5. General lessons for unknown raw formats

1. **Factor the file size.** Integer structure (x2, x10, image dims) is the skeleton.
2. **Hexdump a dark frame.** Constant patterns expose headers, block layout, bit packing.
3. **Use the camera's own logs.** Black level, exposure, bit depth were all in `sens_*.csv`.
4. **Exploit redundancy.** Bayer greens must match; neighbouring rows must correlate;
   a second camera photographing the same instant is ground truth.
5. **Check for hidden pages/planes** before declaring data lost — `Image.seek()`
   costs one line and in this project it recovered an entire colour sensor.
"""