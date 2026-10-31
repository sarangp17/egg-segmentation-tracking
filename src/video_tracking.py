"""Task 4: video segmentation (trial V3) + simple centroid tracker with persistent IDs.

Video-specific settings were tuned on only 4 frames, so treat them as an assumption of THIS video.
"""
import cv2
import numpy as np
from .classical import clean_mask, filter_blobs

ROI_TOP, ROI_BOTTOM = 160, 440      # eggs were only seen between these image rows
VIDEO_PARAMS = {"core_val": 170, "grow": 21}
TRACKER_SETTINGS = {"max_dist": 100, "extra_dist_per_miss": 5, "max_missed": 24,
                    "min_hits": 3, "ignore_near": 40}
COLOURS = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255), (255, 128, 0)]


def segment_video(frame_bgr, params=None):
    """V3: bright low-saturation core -> opening -> small grow limited to a looser colour rule."""
    p = dict(VIDEO_PARAMS)
    if params:
        p.update(params)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    loose = (sat <= 100) & (val >= 120)
    core = (sat <= 60) & (val >= p["core_val"])
    for m in (loose, core):
        m[:ROI_TOP, :] = False
        m[ROI_BOTTOM:, :] = False
    core = clean_mask(core, 9, 0)
    if p["grow"] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["grow"], p["grow"]))
        grown = cv2.dilate(core.astype(np.uint8), kernel) > 0
    else:
        grown = core
    return filter_blobs(grown & loose, 0.01, 1.0, 3.0)


def find_detections(mask):
    """Centroid, box, area and outline for every blob in the mask."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    dets = []
    for k in range(1, n):
        comp = (labels == k).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets.append({"cx": float(centroids[k][0]), "cy": float(centroids[k][1]),
                     "box": (int(stats[k, 0]), int(stats[k, 1]), int(stats[k, 2]), int(stats[k, 3])),
                     "area": int(stats[k, cv2.CC_STAT_AREA]), "contours": contours})
    return dets


class EggTracker:
    """Greedy nearest-centroid matching with persistent IDs.

    - closest (track, blob) pairs are matched first, within a distance limit that grows
      by extra_dist_per_miss for every frame a track was unseen
    - a new track gets an ID after min_hits frames (eggs in the first frame get one at once)
    - a lost track is kept for max_missed frames (a returning egg keeps its ID if close enough)
    - a new blob very close to an already matched blob is ignored (likely a fragment)
    """

    def __init__(self, max_dist=100, extra_dist_per_miss=5, max_missed=24, min_hits=3, ignore_near=40):
        self.max_dist = max_dist
        self.extra = extra_dist_per_miss
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.ignore_near = ignore_near
        self.tracks = []
        self.next_id = 1

    def update(self, dets, first_frame=False):
        """Process one frame. Returns [(id, detection, trail)] for confirmed, visible tracks."""
        pairs = []
        for ti in range(len(self.tracks)):
            for di in range(len(dets)):
                d = np.hypot(self.tracks[ti]["cx"] - dets[di]["cx"], self.tracks[ti]["cy"] - dets[di]["cy"])
                if d <= self.max_dist + self.extra * self.tracks[ti]["missed"]:
                    pairs.append((d, ti, di))
        pairs.sort()
        used_tracks, used_dets = set(), set()
        for d, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            t = self.tracks[ti]
            t["cx"], t["cy"], t["det"] = dets[di]["cx"], dets[di]["cy"], dets[di]
            t["missed"] = 0
            t["hits"] += 1
            t["trail"].append((int(t["cx"]), int(t["cy"])))
            if t["id"] is None and t["hits"] >= self.min_hits:
                t["id"] = self.next_id
                self.next_id += 1
        for ti in range(len(self.tracks)):
            if ti not in used_tracks:
                self.tracks[ti]["missed"] += 1
                self.tracks[ti]["det"] = None
        for di in range(len(dets)):
            if di in used_dets:
                continue
            too_close = False
            for dj in used_dets:
                if np.hypot(dets[di]["cx"] - dets[dj]["cx"], dets[di]["cy"] - dets[dj]["cy"]) < self.ignore_near:
                    too_close = True
            if too_close:
                continue
            first_id = None
            if first_frame:
                first_id = self.next_id
                self.next_id += 1
            self.tracks.append({"id": first_id, "cx": dets[di]["cx"], "cy": dets[di]["cy"], "missed": 0,
                                "hits": 1, "trail": [(int(dets[di]["cx"]), int(dets[di]["cy"]))], "det": dets[di]})
        keep = []
        for t in self.tracks:
            if t["id"] is None and t["missed"] > 2:
                continue
            if t["id"] is not None and t["missed"] > self.max_missed:
                continue
            keep.append(t)
        self.tracks = keep
        return [(t["id"], t["det"], t["trail"]) for t in self.tracks
                if t["id"] is not None and t["missed"] == 0 and t["det"] is not None]


def draw_tracks(frame, visible, time_s):
    shown = frame.copy()
    for egg_id, det, trail in visible:
        colour = COLOURS[(egg_id - 1) % len(COLOURS)]
        x, y, bw, bh = det["box"]
        cv2.drawContours(shown, det["contours"], -1, colour, 2)
        cv2.rectangle(shown, (x, y), (x + bw, y + bh), colour, 2)
        cv2.circle(shown, (int(det["cx"]), int(det["cy"])), 5, colour, -1)
        if len(trail) > 1:
            cv2.polylines(shown, [np.array(trail[-40:], dtype=np.int32)], False, colour, 2)
        cv2.putText(shown, "egg " + str(egg_id), (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(shown, "t = %.1f s   eggs tracked: %d" % (time_s, len(visible)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return shown


def process_video(video_path, out_path, start_sec, end_sec):
    """Segment + track every frame in [start_sec, end_sec) and write an annotated MP4.
    The notebook additionally re-encodes to H.264 with ffmpeg so Google Drive plays it."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    start_frame, end_frame = int(start_sec * fps), int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    tracker = EggTracker(**TRACKER_SETTINGS)
    ids_seen = {}
    tracked_per_frame = []
    for frame_no in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        visible = tracker.update(find_detections(segment_video(frame)), first_frame=(frame_no == start_frame))
        for egg_id, det, trail in visible:
            ids_seen[egg_id] = ids_seen.get(egg_id, 0) + 1
        tracked_per_frame.append(len(visible))
        writer.write(draw_tracks(frame, visible, frame_no / fps))
    cap.release()
    writer.release()
    return {"frames": len(tracked_per_frame), "frames_seen_per_id": ids_seen,
            "tracked_per_frame": tracked_per_frame}
