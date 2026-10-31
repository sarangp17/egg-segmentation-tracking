"""Final classical segmentation (experiment E6): HSV colour rule -> opening -> blob filter."""
import cv2
import numpy as np

FINAL_PARAMS = {
    "sat_max": 40,        # eggs are white: low saturation
    "val_min": 170,       # ... and bright
    "open_k": 9,          # opening kernel (removes specks and thin bridges)
    "close_k": 0,         # closing was tested (E3) and hurt, so it is off
    "min_area_frac": 0.02,
    "max_edge": 1.0,      # 1.0 = border-contact filter off (it hurt in E4)
    "max_aspect": 3.0,    # touching eggs form long blobs (E6)
}


def clean_mask(mask_bool, open_k, close_k):
    """Morphological opening then closing with elliptical kernels (0 = skip)."""
    m = mask_bool.astype(np.uint8) * 255
    if open_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    if close_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m > 0


def filter_blobs(mask_bool, min_area_frac, max_edge, max_aspect):
    """Keep only connected blobs that are big enough, not wrapped around the
    image border, and not too long and thin."""
    h, w = mask_bool.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8))
    out = np.zeros((h, w), dtype=bool)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area_frac * h * w:
            continue
        comp = labels == k
        border = comp[0, :].sum() + comp[-1, :].sum() + comp[:, 0].sum() + comp[:, -1].sum()
        if border / (2 * (h + w)) > max_edge:
            continue
        contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = max(contours, key=cv2.contourArea)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(biggest)
        short_side = min(rw, rh)
        aspect = max(rw, rh) / short_side if short_side > 0 else 99
        if aspect > max_aspect:
            continue
        out[comp] = True
    return out


def segment_eggs(img_bgr, params=None):
    """Return a boolean egg mask for one BGR image."""
    p = dict(FINAL_PARAMS)
    if params:
        p.update(params)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] <= p["sat_max"]) & (hsv[:, :, 2] >= p["val_min"])
    mask = clean_mask(mask, p["open_k"], p["close_k"])
    return filter_blobs(mask, p["min_area_frac"], p["max_edge"], p["max_aspect"])
