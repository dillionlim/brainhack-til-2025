import base64
import json
import math
import os
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
import requests
import itertools
from tqdm import tqdm
import numpy as np


TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")
BATCH_SIZE = 4


def sample_generator(
        instance_dirs: Sequence[Path],
) -> Iterator[Mapping[str, Any]]:
    for instance_dir in instance_dirs:
        slices_dir = instance_dir / "slices"

        slices = []
        for slice_idx in range(len(list(slices_dir.iterdir()))):
            with open(slices_dir / f"{slice_idx}.jpg", "rb") as slice_file:
                slice_bytes = slice_file.read()
            slices.append(base64.b64encode(slice_bytes).decode("ascii"))
        
        yield {"key": int(instance_dir.stem), "slices": slices}


def is_permutation(seq: Sequence[int]):
    return all(i in seq for i in range(len(seq)))


def correct_subsegment_lengths(
        gt: Sequence[int], pred: Sequence[int],
) -> Sequence[int]:
    assert len(pred) == len(gt) and is_permutation(pred) and is_permutation(gt)
    
    n = len(gt)
    gt_indices = [None] * n
    for i, y in enumerate(gt):
        gt_indices[y] = i

    lens = []
    curr_correct_len = 1
    for i in range(1, n):
        if gt_indices[pred[i]] == gt_indices[pred[i-1]] + 1:
            curr_correct_len += 1
        else:
            lens.append(curr_correct_len)
            curr_correct_len = 1
    lens.append(curr_correct_len)

    return lens


def entropy(seq: np.ndarray, base: float) -> np.float64:
    return -np.sum(seq * np.log2(seq) / np.log2(base))


def score_surprise(preds: Sequence[Sequence[int]], ground_truth: Sequence[Sequence[int]]) -> float:
    avg_score = 0.
    for gt, pred in zip(ground_truth, preds):
        lengths = np.array(correct_subsegment_lengths(gt, pred))
        this_instance_score = 1 - entropy(lengths / len(gt), base=len(gt))
        avg_score += this_instance_score.item() / len(ground_truth)

    return avg_score


def main():
    # Testing on GCP
    # data_dir = Path.home() / TEAM_TRACK / "surprise"
    # results_dir = Path.home() / TEAM_NAME
    # results_dir.mkdir(parents=True, exist_ok=True)

    # For local testing
    data_dir = Path("./data")
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)

    instance_dirs = list(data_dir.iterdir())[:45]
    batch_generator = itertools.batched(sample_generator(instance_dirs), n=BATCH_SIZE)

    results = []
    for batch in tqdm(batch_generator, total=math.ceil(len(instance_dirs) / BATCH_SIZE)):      
        response = requests.post("http://localhost:5005/surprise", data=json.dumps({
            "instances": batch,
        }))
        results.extend(response.json()["predictions"])

    results_path = results_dir / "surprise_results.json"
    print(f"Saving test results to {str(results_path)}")
    with open(results_path, "w") as results_file:
        json.dump(results, results_file)
    
    ground_truths = []
    for instance_dir in instance_dirs:
        with open(instance_dir / "gt.txt", "r") as gt_file:
            ground_truths.append(list(map(int, gt_file.read().split(" "))))

    score = score_surprise(results, ground_truths)
    print("Average score:", score)


if __name__ == "__main__":
    main()
