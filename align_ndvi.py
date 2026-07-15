"""Align cam2 (RED/NIR) to cam1 (RGB) for the Sony MSZ-2100G and overlay NDVI.

Usage:
    python align_ndvi.py <session_dir> <out_dir> [--calib calib.json]

For each frame pair in the session:
  1. decode RGB (fw2.00 .raw or fw2.20 4-page .tiff) at half-res colour (2272x1292)
  2. estimate cam2->cam1 homography by ECC on gradient images (RED band vs R channel)
     - reuses/saves the base homography in calib.json (rotation/scale are stable)
  3. refine the translation PER FRAME (fast ECC at 1/4 scale) — this tracks the
     parallax shift caused by camera height/distance changes (3 cm lens baseline)
  4. compute NDVI from RED/NIR, warp into the RGB frame, alpha-blend with the
     standard red(low) -> yellow -> green(high) colormap.

Notes:
  - RED and NIR come from the same sensor, so NDVI needs no internal alignment.
  - Frames with too little texture for refinement fall back to the base transform.
  - RGB white balance is chart-calibrated for daylight (see wb_daylight.json);
    pass wb=None to render_rgb for scene-adaptive gray-world instead.
"""
import numpy as np, cv2, sys, os, glob, json
from PIL import Image

BL2 = 239.0   # cam2 black level (12-bit DN)

def load_band(p):
    return np.fromfile(p, dtype='<u2').reshape(540, 960).astype(np.float32)/16 - BL2

def load_rgb(path):
    if path.endswith('.tiff'):
        im = Image.open(path); pl=[]
        for k in range(4):
            im.seek(k); pl.append(np.asarray(im).astype(np.float32))
        G=(pl[0]+pl[3])/2
        return np.stack([pl[1]-15, G-15, pl[2]-15], -1).clip(0)
    b = np.fromfile(path, dtype=np.uint8).reshape(-1,10)
    base, shift = b[:,0].astype(np.int32), b[:,1].astype(np.int32)
    nib = np.empty((len(b),16), np.int32)
    nib[:,0::2]=b[:,2:]&0xF; nib[:,1::2]=b[:,2:]>>4
    px = ((base[:,None]<<4)+(nib<<(4+shift[:,None]))).reshape(1292,568,16)
    ev = px[:,:284].reshape(1292,142,2,16).transpose(0,1,3,2).reshape(1292,4544)
    od = px[:,284:].reshape(1292,142,2,16).transpose(0,1,3,2).reshape(1292,4544)
    m = np.empty((2584,4544), np.float32); m[0::2]=ev; m[1::2]=od
    R=m[0::2,0::2]-BL2; G=(m[0::2,1::2]+m[1::2,0::2])/2-BL2; B=m[1::2,1::2]-BL2
    return np.stack([R,G,B],-1).clip(0)

def grad(a):
    a = cv2.GaussianBlur(a, (5,5), 1.2)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0); gy = cv2.Sobel(a, cv2.CV_32F, 0, 1)
    g = np.sqrt(gx*gx+gy*gy)
    lo, hi = np.percentile(g, [50, 99.5])
    return np.clip((g-lo)/(hi-lo+1e-6), 0, 1).astype(np.float32)

def estimate_h(red, rgb):
    """Pyramid ECC: affine at 1/4 scale of the cam1 half-res frame, refined at 1/2.
    Returns H mapping cam2 (960x540) coords into cam1 half-res frame (2272x1292)."""
    tgt_full = grad(rgb[...,0])                  # R channel ~ RED band spectrally
    src = grad(red)                              # 960x540
    cc_out = -1.0
    W = np.eye(2,3, dtype=np.float32)
    for sc, iters in ((4, 120), (2, 60)):
        tgt = cv2.resize(tgt_full, (2272//sc, 1292//sc), interpolation=cv2.INTER_AREA)
        srcs = cv2.resize(src, (2272//sc, int(540*2272/960)//sc), interpolation=cv2.INTER_AREA)
        if sc != 4:
            W[:, 2] *= 2.0       # translations scale with resolution
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
        try:
            cc_out, W = cv2.findTransformECC(tgt, srcs, W, cv2.MOTION_AFFINE, crit, None, 3)
        except cv2.error:
            pass
    sc = 2
    S_cam2_to_srcs = np.array([[ (2272//sc)/960., 0, 0],
                               [0, (int(540*2272/960)//sc)/540., 0],
                               [0,0,1]], np.float32)
    W3 = np.vstack([W, [0,0,1]]).astype(np.float32)
    S_lvl_to_full = np.array([[sc,0,0],[0,sc,0],[0,0,1]], np.float32)
    H = S_lvl_to_full @ np.linalg.inv(W3) @ S_cam2_to_srcs
    return H.astype(np.float32), float(cc_out)

def refine_translation(red, rgb, H_base):
    """Per-frame parallax correction: warp RED by H_base, then estimate the
    residual translation against the RGB red channel with ECC at 1/4 scale.
    Returns H with updated translation and the ECC score."""
    sc = 4
    tgt = cv2.resize(grad(rgb[...,0]), (2272//sc, 1292//sc), interpolation=cv2.INTER_AREA)
    warped = cv2.warpPerspective(red, H_base, (2272,1292))
    src = cv2.resize(grad(warped), (2272//sc, 1292//sc), interpolation=cv2.INTER_AREA)
    W = np.eye(2,3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
    try:
        cc, W = cv2.findTransformECC(tgt, src, W, cv2.MOTION_TRANSLATION, crit, None, 3)
    except cv2.error:
        return H_base, -1.0
    # shift in full-res half-frame coords (invert: W maps tgt->src)
    T = np.array([[1,0,-W[0,2]*sc],[0,1,-W[1,2]*sc],[0,0,1]], np.float32)
    return (T @ H_base).astype(np.float32), float(cc)

def ndvi_map(red, nir):
    r = cv2.blur(red, (3,3)).clip(0.1); n = cv2.blur(nir, (3,3)).clip(0.1)
    ndvi = (n-r)/(n+r)
    valid = (n+r) > 30
    return ndvi, valid

def colormap_ryg(ndvi, lo=-0.5, hi=0.5):
    """red (low) -> yellow -> green (high)"""
    t = np.clip((ndvi-lo)/(hi-lo), 0, 1)
    c = np.zeros((*ndvi.shape,3), np.float32)
    c[...,0] = np.clip(2*(1-t), 0, 1)   # R
    c[...,1] = np.clip(2*t, 0, 1)       # G
    return c

# Daylight white balance measured from a colour chart's neutral patches
# (session 20170701000045). Pass wb=None for scene-adaptive gray-world.
WB_DAYLIGHT = np.array([1.810, 1.0, 1.641], np.float32)
SAT = 240.0   # channel saturation level after black subtraction (fw2.20 planes)

def render_rgb(c, wb=WB_DAYLIGHT):
    """Display render: chart-calibrated WB (or gray-world), highlight-safe
    clipping, tone curve, saturation, chroma denoise, unsharp mask."""
    H, W, _ = c.shape
    bm = np.median(c[:H//4*4, :W//4*4].reshape(H//4,4,W//4,4,3), axis=(1,3))
    if wb is None:
        med = np.median(bm.reshape(-1,3), axis=0)+1e-6
        w99 = np.percentile(bm.reshape(-1,3), 99, axis=0)+1e-6
        wb = 0.5*(med.mean()/med) + 0.5*(w99.mean()/w99)
    c = np.minimum(c * wb, SAT)   # clip at green-saturation point: blown = white
    bm = np.minimum(bm * wb, SAT)
    lum = c.mean(axis=2)
    lo, hi = np.percentile(lum, [0.5, 99.5])
    if hi - lo < 6: hi = lo + 30          # near-black guard
    t = np.clip((c-lo)/(hi-lo), 0, 1)**0.55
    y = t.mean(axis=2, keepdims=True)
    t = np.clip(y + (t-y)*1.35, 0, 1)     # saturation
    img = (t*255).astype(np.uint8)
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycc[...,1] = cv2.medianBlur(ycc[...,1], 5)
    ycc[...,2] = cv2.medianBlur(ycc[...,2], 5)
    yl = ycc[...,0].astype(np.float32)
    ycc[...,0] = np.clip(yl + 0.7*(yl - cv2.GaussianBlur(yl,(0,0),1.5)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)

def main():
    sdir, odir = sys.argv[1], sys.argv[2]
    calib = sys.argv[sys.argv.index('--calib')+1] if '--calib' in sys.argv else None
    os.makedirs(odir, exist_ok=True)
    H_saved = None
    if calib and os.path.exists(calib):
        H_saved = np.array(json.load(open(calib))['H'], np.float32)
    rgbs = sorted(glob.glob(os.path.join(sdir,'cam1_*_RGB.tiff')) +
                  glob.glob(os.path.join(sdir,'cam1_*_RGB.raw')))
    for rp in rgbs:
        idx = os.path.basename(rp).split('_')[-2]
        base = '_'.join(os.path.basename(rp).split('_')[:2]).replace('cam1','cam2')
        rf = os.path.join(sdir, f"{base}_{idx}_RED.raw")
        nf = os.path.join(sdir, f"{base}_{idx}_NIR.raw")
        if not (os.path.exists(rf) and os.path.exists(nf)): continue
        rgb = load_rgb(rp); red, nir = load_band(rf), load_band(nf)
        if H_saved is None:
            H0, cc0 = estimate_h(red, rgb)
            if cc0 > 0.2 and calib:
                json.dump({'H': H0.tolist(), 'ecc': cc0}, open(calib,'w'), indent=1)
                H_saved = H0
        else:
            H0 = H_saved
        # per-frame parallax refinement (translation tracks camera height)
        H, cc = refine_translation(red, rgb, H0)
        if cc < 0.15:                      # low texture: keep base transform
            H = H0
        ndvi, valid = ndvi_map(red, nir)
        ndvi_w = cv2.warpPerspective(ndvi, H, (2272,1292), flags=cv2.INTER_LINEAR)
        valid_w = cv2.warpPerspective(valid.astype(np.float32), H, (2272,1292)) > 0.5
        base_img = render_rgb(rgb).astype(np.float32)/255
        cmap = colormap_ryg(ndvi_w)
        alpha = 0.45*valid_w[...,None]
        out = (base_img*(1-alpha) + cmap*alpha)
        Image.fromarray((out*255).astype(np.uint8)).save(
            os.path.join(odir, f"overlay_{idx}.png"))
        print(f"{idx}: ecc={cc:.3f}  ndvi median {np.median(ndvi[valid]) if valid.any() else float('nan'):.3f}")

if __name__ == '__main__':
    main()
