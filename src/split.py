"""Leakage-aware 80:20 split: dHash fingerprints -> near-duplicate groups -> split whole groups."""
import cv2
import numpy as np


def dhash(img_bgr):
    """64-bit difference hash: shrink to 9x8 grey and compare each pixel to its right neighbour."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8))
    return (small[:, 1:] > small[:, :-1]).flatten()


def group_near_duplicates(hashes, threshold=10):
    """Union-find grouping: images whose hashes differ in <= threshold bits share a group.
    Returns a list of groups (each group is a list of image indices)."""
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if int(np.sum(hashes[i] != hashes[j])) <= threshold:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def split_groups(groups, n_images, seed=42, eval_fraction=0.2):
    """Move whole groups to the evaluation set (seeded shuffle) until it holds about eval_fraction."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups))
    target_eval = int(round(eval_fraction * n_images))
    eval_idx = []
    for k in order:
        if len(eval_idx) + len(groups[k]) <= target_eval:
            eval_idx = eval_idx + groups[k]
    eval_set = set(eval_idx)
    dev_idx = [i for i in range(n_images) if i not in eval_set]
    return dev_idx, sorted(eval_idx)
