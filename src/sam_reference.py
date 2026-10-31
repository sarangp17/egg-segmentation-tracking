"""Task 1: turn SAM proposals into one binary egg mask per image (the pseudo-ground truth).

SAM (automatic mode) proposes many regions. We keep a region only if it passes simple
checks (size, smoothness, shape, border contact, colour) and merge the kept regions.
Images the rule cannot handle get manual box prompts (disclosed in the README).
"""
import cv2
import numpy as np

RULE = {
    "min_area_frac": 0.02, "max_area_frac": 0.90,
    "min_solidity": 0.88, "max_aspect": 2.5,
    "max_edge": 0.20, "max_sat": 70,
    "min_val_strict": 170,    # step 1
    "min_val_relaxed": 155,   # step 2, only if step 1 found nothing
}

SAM_SETTINGS = {"points_per_side": 32, "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.92, "min_mask_region_area": 200}


def load_sam(checkpoint_path, device="cuda"):
    """Load SAM ViT-B. Imported here so the rest of the module works without torch."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device)
    generator = SamAutomaticMaskGenerator(sam, **SAM_SETTINGS)
    return generator, SamPredictor(sam)


def region_features(mask, hsv, img_area):
    """Simple numbers describing one SAM region."""
    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape
    area = int(mask_u8.sum())
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    biggest = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(biggest))
    solidity = cv2.contourArea(biggest) / hull_area if hull_area > 0 else 0
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(biggest)
    short_side = min(rw, rh)
    aspect = max(rw, rh) / short_side if short_side > 0 else 99
    border = mask_u8[0, :].sum() + mask_u8[-1, :].sum() + mask_u8[:, 0].sum() + mask_u8[:, -1].sum()
    return {"area_frac": area / img_area, "solidity": solidity, "aspect": aspect,
            "edge": border / (2 * (h + w)),
            "sat": float(hsv[:, :, 1][mask].mean()), "val": float(hsv[:, :, 2][mask].mean())}


def is_egg(f, min_val, rule=RULE):
    """True if the region passes every check."""
    if f is None:
        return False
    if f["area_frac"] < rule["min_area_frac"] or f["area_frac"] > rule["max_area_frac"]:
        return False
    if f["solidity"] < rule["min_solidity"] or f["aspect"] > rule["max_aspect"]:
        return False
    if f["edge"] > rule["max_edge"]:
        return False
    if f["sat"] > rule["max_sat"] or f["val"] < min_val:
        return False
    return True


def fill_holes(mask_bool):
    """Fill holes (dirt spots, reflections) inside the mask."""
    m = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    return filled.astype(bool)


def pick_union(sam_masks, hsv, img_area, min_val, rule=RULE):
    """Merge all SAM regions that pass the rule. Returns (union mask, how many passed)."""
    union = np.zeros(hsv.shape[:2], dtype=bool)
    count = 0
    for m in sam_masks:
        f = region_features(m["segmentation"], hsv, img_area)
        if is_egg(f, min_val, rule):
            union = union | m["segmentation"]
            count += 1
    return union, count


def biggest_piece(mask_bool):
    """Keep only the largest connected piece (removes stray specks)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8))
    if n <= 1:
        return mask_bool
    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == best


def box_mask(predictor, box, h, w):
    """SAM mask for one box (left, top, right, bottom), cut to the box, biggest piece only.
    predictor.set_image(img_rgb) must have been called first."""
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(w, box[2]), min(h, box[3])
    out, scores, _ = predictor.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
    clipped = np.zeros_like(out[0])
    clipped[y1:y2, x1:x2] = out[0][y1:y2, x1:x2]
    return biggest_piece(clipped)


def build_reference_mask(img_rgb, generator, predictor, only_boxes=None, extra_boxes=None):
    """Final reference mask for one image. Returns (mask, method).

    only_boxes  -> manual boxes only ("manual_box")
    extra_boxes -> automatic rule plus boxes for eggs the rule missed ("..._plus_box")
    """
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    if only_boxes:
        predictor.set_image(img_rgb)
        union = np.zeros((h, w), dtype=bool)
        for b in only_boxes:
            union = union | box_mask(predictor, b, h, w)
        return fill_holes(union), "manual_box"
    sam_masks = generator.generate(img_rgb)
    union, count = pick_union(sam_masks, hsv, h * w, RULE["min_val_strict"])
    method = "auto_strict"
    if count == 0:
        union, count = pick_union(sam_masks, hsv, h * w, RULE["min_val_relaxed"])
        method = "auto_relaxed"
    if extra_boxes:
        predictor.set_image(img_rgb)
        for b in extra_boxes:
            union = union | box_mask(predictor, b, h, w)
        method += "_plus_box"
    return fill_holes(union), method
