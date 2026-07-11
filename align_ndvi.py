"""Align cam2 (RED/NIR) to cam1 (RGB) for the Sony MSZ-2100G and overlay NDVI.

Usage:
    python align_ndvi.py <session_dir> <out_dir> [--calib calib.json]

For each frame pair in the session:
  1. decode RGB (fw2.00 .raw or fw2.20 4-page .tiff) at half-res colour (2272x1292)
  2. estimate cam2->cam1 homography by ECC on gradient images (RED band vs R channel)
     - reuses/saves the homography in calib.json (per-camera it is stable at distance)
  3. compute NDVI from RED/NIR, warp into the RGB frame, alpha-blend with the
     standard red(low) -> yellow -> green(high) colormap.

Notes:
  - RED and NIR come from the same sensor, so NDVI needs no internal alignment;
    only the cam2->cam1 transform is estimated.
  - The two lenses sit ~3 cm apart: at drone altitude one saved transform serves
    all frames; at close range expect parallax off the dominant scene plane.
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

def render_rgb(c):
    bm = np.median(c[:1288,:2272].reshape(322,4,568,4,3), axis=(1,3))
    med = np.median(bm.reshape(-1,3), axis=0)+1e-6
    c = c*np.clip(med.mean()/med, 0.7, 1.5)
    hi = max(np.percentile(bm.mean(axis=2), 99.8), 30)
    return (np.clip(c/hi, 0, 1)**0.5*255).astype(np.uint8)

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
        if H_saved is not None:
            H, cc = H_saved, float('nan')
        else:
            H, cc = estimate_h(red, rgb)
            if cc > 0.2 and calib:
                json.dump({'H': H.tolist(), 'ecc': cc}, open(calib,'w'), indent=1)
                H_saved = H
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
