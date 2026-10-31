"""Smoke tests on SYNTHETIC data (no dataset, no GPU, no SAM needed).
Run from the repo root:  python tests/smoke_test.py
They check that the code runs and behaves sensibly. They do NOT reproduce the reported scores."""
import os, sys, tempfile
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import compute_metrics, summarise
from src.classical import segment_eggs
from src.split import dhash, group_near_duplicates, split_groups
from src.sam_reference import region_features, is_egg, fill_holes
from src.video_tracking import process_video


def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    if not condition:
        sys.exit(1)


# 1. metrics
a = np.zeros((10, 10), dtype=bool); a[2:6, 2:6] = True
check("metrics: identical masks give IoU 1", compute_metrics(a, a)["iou"] == 1.0)
check("metrics: disjoint masks give IoU 0", compute_metrics(a, np.roll(a, 5, axis=0))["iou"] == 0.0)
empty = np.zeros((10, 10), dtype=bool)
check("metrics: both empty gives 1", compute_metrics(empty, empty)["iou"] == 1.0)
check("metrics: predicted nothing, egg expected -> recall 0", compute_metrics(empty, a)["recall"] == 0.0)
s = summarise([compute_metrics(a, a), compute_metrics(empty, a)])
check("summarise: macro IoU is the mean", abs(s["macro_iou"] - 0.5) < 1e-9)

# 2. classical segmentation on a synthetic egg (white ellipse on a darker background)
img = np.full((640, 640, 3), 90, dtype=np.uint8)
truth = np.zeros((640, 640), dtype=np.uint8)
cv2.ellipse(truth, (320, 320), (120, 160), 0, 0, 360, 255, -1)
img[truth > 0] = (245, 245, 245)
pred = segment_eggs(img)
iou = compute_metrics(pred, truth > 0)["iou"]
check("classical: synthetic egg IoU > 0.9 (got %.3f)" % iou, iou > 0.9)
check("classical: plain background gives no egg", segment_eggs(np.full((640, 640, 3), 90, dtype=np.uint8)).sum() == 0)

# 3. SAM-rule helpers
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
f = region_features(truth > 0, hsv, 640 * 640)
check("sam rule: a bright oval is accepted", is_egg(f, 170))
whole = np.ones((640, 640), dtype=bool)
check("sam rule: a mask covering the whole image is rejected", not is_egg(region_features(whole, hsv, 640 * 640), 170))
ring = (truth > 0).copy(); ring[300:340, 300:340] = False
check("fill_holes fills a hole inside the egg", fill_holes(ring)[320, 320])

# 4. split
rng = np.random.RandomState(0)
hashes = [rng.rand(64) > 0.5 for _ in range(110)]
hashes[1] = hashes[0].copy(); hashes[2] = hashes[0].copy()      # three identical images
groups = group_near_duplicates(hashes, threshold=10)
dev, ev = split_groups(groups, 110, seed=42)
check("split: sizes add up to 110 and eval is about 20%%  (dev %d / eval %d)" % (len(dev), len(ev)),
      len(dev) + len(ev) == 110 and 20 <= len(ev) <= 22)
check("split: no image in both sets", len(set(dev) & set(ev)) == 0)
check("split: identical images stay in the same set", len({0, 1, 2} & set(ev)) in (0, 3))
check("split: deterministic with the same seed", split_groups(groups, 110, seed=42) == (dev, ev))
check("dhash: 64 bits", dhash(img).shape == (64,))

# 5. tracker on a synthetic video: two white circles moving, zone y=160..440
tmp = tempfile.mkdtemp()
video = os.path.join(tmp, "synthetic.mp4")
writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 24, (854, 480))
for i in range(48):
    frame = np.full((480, 854, 3), 60, dtype=np.uint8)
    cv2.circle(frame, (150 + 4 * i, 300), 50, (255, 255, 255), -1)
    cv2.circle(frame, (650 - 2 * i, 330), 50, (255, 255, 255), -1)
    writer.write(frame)
writer.release()
stats = process_video(video, os.path.join(tmp, "out.mp4"), 0, 2)
check("tracker: 2 IDs for 2 moving eggs (got %s)" % list(stats["frames_seen_per_id"].keys()),
      len(stats["frames_seen_per_id"]) == 2)
check("tracker: both eggs tracked in most frames (frames with 2 eggs: %d of %d)" %
      (sum(1 for c in stats["tracked_per_frame"] if c == 2), stats["frames"]),
      sum(1 for c in stats["tracked_per_frame"] if c == 2) >= 44)
check("tracker: annotated video was written", os.path.getsize(os.path.join(tmp, "out.mp4")) > 1000)

print("\nAll smoke tests passed.")
