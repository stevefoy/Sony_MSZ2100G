"""Sony MSZ-2100G raw converter.

Usage:
    python msz_convert.py <raw_root> <output_root>

Inputs handled (see docs/FORMAT.md for full reverse-engineered specs):
  cam2 (*_RED.raw / *_NIR.raw)  960x540, uint16 LE, 12-bit DN << 4, black 239.
  cam1 (*_RGB.raw, fw 2.00)     4544x2584 Bayer RGGB, Sony on-sensor block
      compression: 10-byte block = base byte + shift byte + 16 x 4-bit deltas;
      pixel = (base<<4) + (nibble << (4+shift)). Stored row = 568 blocks:
      first 284 -> even sensor row as alternating [16xR][16xGr] blocks,
      last 284  -> odd sensor row as alternating [16xGb][16xB] blocks.
  cam1 (*_RGB.tiff, fw 2.20)    4-page TIFF = Bayer planes Gr,R,B,Gb,
      each 2272x1292 8-bit JPEG, black ~15.

Outputs per session: png/ previews, tif/ 16-bit sensor values, ndvi/ maps.
Existing outputs are skipped (safe to re-run incrementally).
"""
import numpy as np, sys, os
from PIL import Image

BL_DBP = 239*16   # cam2: 12-bit data stored <<4
BL_RGB = 239      # cam1 fw2.00: decoded 12-bit DN
BL_TIFF = 15      # cam1 fw2.20: 8-bit planes

def stretch(a, plo=0.5, phi=99.7, gamma=0.6):
    lo, hi = np.percentile(a, [plo, phi])
    if hi - lo < 8: hi = lo + 8
    return (np.clip((a-lo)/(hi-lo), 0, 1)**gamma * 255).astype(np.uint8)

def decode_dbp(path):
    return np.fromfile(path, dtype='<u2').reshape(540, 960)

def decode_rgb(path):
    b = np.fromfile(path, dtype=np.uint8).reshape(-1, 10)
    base  = b[:,0].astype(np.int32)   # block base = min >> 4
    shift = b[:,1].astype(np.int32)   # adaptive range shift (0..2 seen)
    nib = np.empty((b.shape[0],16), np.int32)
    nib[:,0::2] = b[:,2:] & 0xF; nib[:,1::2] = b[:,2:] >> 4
    px = ((base[:,None]<<4) + (nib << (4 + shift[:,None]))).reshape(1292, 568, 16)
    ev = px[:, :284].reshape(1292,142,2,16).transpose(0,1,3,2).reshape(1292,4544)
    od = px[:, 284:].reshape(1292,142,2,16).transpose(0,1,3,2).reshape(1292,4544)
    img = np.empty((2584,4544), np.int32); img[0::2]=ev; img[1::2]=od
    return img

def demosaic(img):
    """simple half-res demosaic RGGB -> HxWx3 float (black-subtracted)"""
    R  = img[0::2,0::2].astype(float) - BL_RGB
    G  = (img[0::2,1::2].astype(float) + img[1::2,0::2].astype(float))/2 - BL_RGB
    B  = img[1::2,1::2].astype(float) - BL_RGB
    return np.stack([R,G,B], -1).clip(0)

def _shift_stack(a):
    H,W = a.shape
    p = np.pad(a, 1, mode='edge')
    return np.stack([p[dy:dy+H, dx:dx+W] for dy in range(3) for dx in range(3)])

def med3(a):
    return np.median(_shift_stack(a), axis=0)

def denoise(c, strength):
    # 1) hot/dead pixel clamp per channel: clip to local min/max of 8 neighbours
    for k in range(3):
        st = _shift_stack(c[...,k])
        nb = np.delete(st, 4, axis=0)
        c[...,k] = np.clip(c[...,k], nb.min(axis=0), nb.max(axis=0))
    if strength <= 0: return c
    # 2) luma/chroma split: median-filter chroma hard, luma gently
    y = c.mean(axis=2)
    ym = med3(y)
    y2 = y + np.clip(ym - y, -strength, strength)   # mild luma smoothing
    cb = c - y[...,None]
    cbm = np.stack([med3(med3(cb[...,k])) for k in range(3)], -1)
    return np.clip(y2[...,None] + cbm, 0, None)

def rgb_png(img, out):
    c = demosaic(img)
    # robust stats via 4x4 block medians (kills hot pixels)
    H, W, _ = c.shape
    bm = np.median(c[:H//4*4, :W//4*4].reshape(H//4,4,W//4,4,3), axis=(1,3))
    lum0 = bm.mean(axis=2)
    lo, hi = np.percentile(lum0, [0.5, 99.8])
    med = np.median(bm.reshape(-1,3), axis=0) + 1e-6
    gain = np.clip(med.mean()/med, 0.7, 1.5)   # bounded gray-world WB
    c = c * gain
    if hi - lo < 24: hi = lo + 400    # near-black frame: render dark, don't amplify noise
    noise_rel = 8.0/(hi-lo)           # heavier denoise for low-signal frames
    c = denoise(c, strength=3 if noise_rel > 0.05 else 1)
    arr = (np.clip((c-lo)/(hi-lo),0,1)**0.45*255).astype(np.uint8)
    Image.fromarray(arr).save(out)
    return hi

def tiff_png(f, out):
    """fw2.20 cam1: reconstruct colour from 4-page TIFF (planes Gr,R,B,Gb)."""
    im = Image.open(f); pl = []
    for k in range(4):
        try:
            im.seek(k); pl.append(np.array(im).astype(float))
        except EOFError:
            break
    if len(pl) == 4:
        G = (pl[0]+pl[3])/2; R, B = pl[1], pl[2]
        c = np.stack([R-BL_TIFF, G-BL_TIFF, B-BL_TIFF], -1).clip(0)
        bm = np.median(c[:1288,:2272].reshape(322,4,568,4,3), axis=(1,3))
        lum = bm.mean(axis=2)
        lo, hi = np.percentile(lum, [0.5, 99.8])
        sig = 'signal' if hi - lo > 12 else 'dark'
        if hi - lo < 6: hi = lo + 30
        med = np.median(bm.reshape(-1,3), axis=0)+1e-6
        c = c * np.clip(med.mean()/med, 0.7, 1.5)
        c = denoise(c, strength=2 if sig=='dark' else 0)
        Image.fromarray((np.clip((c-lo)/(hi-lo),0,1)**0.5*255).astype(np.uint8)).save(out)
    else:  # single-page fallback
        a = pl[0]
        lo, hi = np.percentile(a, [0.5, 99.8])
        sig = 'signal' if hi - lo > 12 else 'dark'
        if hi - lo < 6: hi = lo + 30
        Image.fromarray((np.clip((a-lo)/(hi-lo),0,1)**0.6*255).astype(np.uint8)).save(out)
    return sig

def box3(a):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    return sw(np.pad(a, 1, mode='edge'), (3, 3)).mean(axis=(2, 3))

def band_has_signal(dn):
    """dn: 12-bit DN image. True if block-mean spread indicates real image content."""
    bm = dn[:536].reshape(67, 8, 120, 8).mean(axis=(1, 3))
    return np.percentile(bm, 99) - np.percentile(bm, 1) > 20

def ndvi_session(sdir, ddir):
    """Compute NDVI for every RED/NIR pair with signal in both bands."""
    import glob
    pairs = sorted(glob.glob(os.path.join(sdir, 'cam2_*_RED.raw')))
    for rf in pairs:
        nf = rf.replace('_RED.raw', '_NIR.raw')
        if not os.path.exists(nf): continue
        idx = os.path.basename(rf).split('_')[-2]
        tp = os.path.join(ddir, f'ndvi_{idx}.tif'); pp = os.path.join(ddir, f'ndvi_{idx}.png')
        if os.path.exists(tp) and os.path.exists(pp): continue
        load = lambda p: np.fromfile(p, dtype='<u2').reshape(540, 960).astype(float)/16 - 239.0
        red, nir = load(rf), load(nf)
        if not (band_has_signal(red) and band_has_signal(nir)):
            print(f"  ndvi_{idx}: skipped (insufficient signal)"); continue
        os.makedirs(ddir, exist_ok=True)
        r, n = box3(red).clip(0.1), box3(nir).clip(0.1)
        ndvi = (n - r) / (n + r); mask = (n + r) < 30
        Image.fromarray(ndvi.astype(np.float32), mode='F').save(tp)
        t = np.clip(ndvi + 0.5, 0, 1)
        rgb = np.zeros((540, 960, 3), np.uint8)
        rgb[..., 0] = np.clip(2*(1-t), 0, 1)*255
        rgb[..., 1] = np.clip(2*t, 0, 1)*255
        rgb[mask] = 90
        Image.fromarray(rgb).save(pp)
        print(f"  ndvi_{idx}: NDVI median %.3f" % np.median(ndvi[~mask]))

def main(src_root, dst_root):
    import glob, shutil
    for sess in sorted(os.listdir(src_root)):
        sdir = os.path.join(src_root, sess)
        if not os.path.isdir(sdir): continue
        for sub in ('png','tif'):
            os.makedirs(os.path.join(dst_root, sess, sub), exist_ok=True)
        for f in sorted(glob.glob(os.path.join(sdir,'*.raw')) + glob.glob(os.path.join(sdir,'*.tiff'))):
            name = os.path.splitext(os.path.basename(f))[0]
            pp = os.path.join(dst_root, sess, 'png', name+'.png')
            tp = os.path.join(dst_root, sess, 'tif', name+'.tif')
            if os.path.exists(pp) and os.path.exists(tp):
                continue
            if f.endswith('.tiff'):
                if not os.path.exists(tp): shutil.copy(f, tp)
                sig = tiff_png(f, pp)
            elif name.endswith('_RGB'):
                img = decode_rgb(f)
                if not os.path.exists(tp):
                    Image.fromarray(img.astype(np.uint16)).save(tp)   # full-res Bayer mosaic DNs
                hi = rgb_png(img, pp)
                sig = 'signal' if hi > 320 else 'dark'
            else:
                img = decode_dbp(f)
                Image.fromarray(img).save(tp)
                Image.fromarray(stretch(img.astype(float))).save(pp)
                sig = 'signal' if img.astype(float).std() > 40 else 'dark'
            print(f"{sess}/{name}: {sig}")
        ndvi_session(sdir, os.path.join(dst_root, sess, 'ndvi'))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
