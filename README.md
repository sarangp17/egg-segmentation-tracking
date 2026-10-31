# Egg segmentation and tracking

Computer vision internship assignment: pretrained SAM reference masks, a classical OpenCV
segmentation compared against them, and egg tracking on a video.

Author: Sarang Palsutkar

**Annotated tracking video (public Google Drive link):**
https://drive.google.com/file/d/1nzwb-UZDESOID0dqhEN9Ej4_-i6Tr7pG/view?usp=sharing

**PDF report:** `report/report.pdf`  (add the file here)

> Where the numbers come from: every score below was produced by the Colab notebook in
> `notebooks/`. The files in `src/` are a cleaned copy of the same logic. They were only
> smoke-tested on synthetic data (`python tests/smoke_test.py`) and were **not** re-run on the real
> dataset, so the notebook is the source of truth for the reported results.

## Results at a glance

| Item | Result |
|---|---|
| Dataset | 110 images, 640x640 JPG (Roboflow Universe `fire-bf7qe/egg-vzhtr`, version 1, CC BY 4.0) |
| Task 1 reference masks | SAM ViT-B + selection rule. 93 of 110 images automatic (84.5%); **17 of 110 (15.5%) used manual box prompts** |
| Split (Task 2) | 88 development / 22 evaluation, seed 42. dHash found no near-duplicates at threshold 10, so group-aware splitting reduced to a seeded random split |
| Final classical method (E6) | HSV colour rule, opening, blob filter |
| Development score (88 images) | macro IoU 0.496, Dice 0.618, precision 0.684, recall 0.677 |
| **Evaluation score (22 images)** | **macro IoU 0.409, Dice 0.525, precision 0.577, recall 0.647; micro IoU 0.280** |
| Egg counting (evaluation) | predicted blob count matched the reference in 12 of 22 images |
| Video tracking (Task 4) | 744 frames (10 s to 41 s). 8 IDs were created for 2 real eggs (IDs change when a hand lifts or hides an egg) |

These scores measure **agreement with SAM-based pseudo-ground truth**, not accuracy against
human labels.

## Repository layout

```
src/            cleaned copies of the logic (metrics, classical, split, SAM reference, tracking)
notebooks/      the Colab notebook that produced all results
configs/        final_params.json (every frozen parameter)
experiments/    experiment_log.csv (all trials, failed ones included), split_manifest.csv
results/        metrics CSVs, plots, reference-mask log, tracking table, reference_masks.zip
report/         PDF report
tests/          smoke_test.py (synthetic data only)
```

## Setup

Tested on Google Colab. SAM needs a GPU (T4 was used). Everything after the reference masks
runs on CPU.

```
git clone https://github.com/sarangp17/egg-segmentation-tracking.git
cd egg-segmentation-tracking
pip install -r requirements.txt
python tests/smoke_test.py
```

Reference environment printed in the Colab session used for the final steps: numpy 2.1.3,
matplotlib 3.10.0. The SAM steps ran in a separate GPU session, so exact torch versions may differ.

SAM checkpoint (not included, 375 MB):
`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

## Dataset

1. Create a free Roboflow account and an API key.
2. Download version 1 in the "folder" export format:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")          # type it at runtime, never commit it
rf.workspace("fire-bf7qe").project("egg-vzhtr").version(1).download("folder", location="egg_dataset")
```

The "folder" format has no boxes or masks, so the SAM masks are the only reference. The images
are not redistributed in this repository. Dataset licence: CC BY 4.0, provided by a Roboflow user.

## Video

Source video: https://www.youtube.com/watch?v=Y4JyV94y2_A (not redistributed). `yt-dlp` was
blocked by YouTube's bot check inside Colab, so the video was downloaded on a laptop and uploaded to Drive
as `video/input.mp4`. Eggs are visible from about 10 s to 41 s; only that part was processed.

## How to reproduce (notebook order)

Each notebook cell starts with a comment header `# ===== CELL N: ... =====`. The notebook also
contains the exploration history, so some cells are superseded. Use these:

| Step | Cell | Purpose |
|---|---|---|
| 1 | CELL 1 | GPU check, Drive mount, dataset download, inspection |
| 2 | CELL 4 | Install SAM, load ViT-B, test on 6 images |
| 3 | CELL 5 (v3) | Egg-selection rule functions |
| 4 | CELL 9 | **Final reference masks** (rule + the 17 manual box prompts) |
| 5 | CELL 10 | dHash grouping and the 80:20 split, writes `split_manifest.csv` |
| 6 | CELL 11 | Metrics, experiment log helpers, E1 baseline (development images only) |
| 7 | CELLS 12 to 16 | E2 to E6 |
| 8 | CELL 17 | **One** final evaluation on the held-out images, metrics and plots |
| 9 | CELLS 21, 23, 24 | Video: V2, V3, tracker and annotated MP4 (needs Cells 11 to 16 first) |
| 10 | CELL 25 | Collect results into the repo folder |

Superseded cells (kept for the history): CELLS 5 (v1, v2), 6, 7, 8, 18 to 20, 22.

## Method summary

**Task 1.** SAM (automatic mode) proposes regions. A region counts as egg only if it passes:
size 2% to 90% of the image, solidity at least 0.88, aspect ratio at most 2.5, border contact at most 0.20, saturation
at most 70, brightness at least 170. If an image has no passing region the brightness limit relaxes to 155.
Passing regions are merged and holes are filled. 17 images needed manual box prompts (translucent
boiling eggs, eggs on foil or in grass, an egg between fingers, a white tray) or an extra box for a missed egg.

**Task 2.** dHash (9x8) fingerprints, Hamming threshold 10, union-find groups, whole groups moved to the
evaluation set with a seeded shuffle. All tuning used the development set only.

| Trial | Change | Dev macro IoU | Decision |
|---|---|---|---|
| E1 | Grayscale + Otsu | 0.345 | baseline |
| E2 | HSV rule (sat <= 40, val >= 170) | 0.424 | keep |
| E3 | + opening, kernel 9 (closing hurt) | 0.447 | keep |
| E4 | + blob filter, min area 2% (border filter hurt) | 0.492 | keep |
| E5 | + watershed | 0.442 | discard |
| E6 | max aspect 3.0 | 0.496 | **final** |

The full log with parameters, motivation and observations is `experiments/experiment_log.csv`.

**Task 3.** Per-image and aggregate IoU, Dice, precision and recall (macro and micro), plus egg-count
agreement. Plots are in `results/plots/`. Empty-mask rules are in `src/metrics.py`.

**Task 4.** Video-specific segmentation (V3): bright low-saturation core, opening, small grow limited to a looser
colour rule, restricted to image rows 160 to 440. Greedy nearest-centroid tracker with persistent IDs
(max jump 100 px, lost tracks kept 24 frames). The video settings were tuned on 4 frames using rough oval
proxies from hand-drawn boxes, so the video scores are not comparable with the dataset scores.

## Known limitations

- The reference masks are SAM output, partly steered by hand (17 of 110 images). They are not human labels, and SAM's own errors count against the classical method.
- The selection rule was tuned by looking at images from the whole dataset (including images that later landed in the evaluation set). The classical algorithm itself was tuned on the development set only.
- The evaluation set is small (22 images) and the IoU spread is large (std 0.27), so the gap between development (0.496) and evaluation (0.409) should not be over-interpreted.
- A threshold method misses the shadowed side of eggs (recall about 0.65) and fails on dim or translucent eggs. White panels and trays are marked as egg because their colour matches.
- Watershed (E5) lowered the score. Over-splitting is the likely cause but was not verified.
- Video: the tracking zone and brightness settings are tuned to this video. Two close eggs can merge into one blob, and an egg hidden or lifted by a hand for more than 1 second gets a new ID when it returns (8 IDs for 2 eggs).

## AI-tool disclosure

I used Claude (Anthropic) as an assistant for planning, writing and debugging code, and drafting this README and the
report. I ran every experiment in Colab myself, supplied the manual box prompts, inspected the outputs and made
the keep/discard decisions. SAM is a pretrained model by Meta AI (Apache-2.0).


## Licences and credits

- Dataset: "egg" by a Roboflow Universe user (`fire-bf7qe`), CC BY 4.0, https://universe.roboflow.com/fire-bf7qe/egg-vzhtr
- SAM: https://github.com/facebookresearch/segment-anything (Apache-2.0)
- Source video on YouTube (link above), not redistributed. The Drive file contains only the annotated segment.
