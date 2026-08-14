"""Flag candidate pretraining images that perceptually match held-out LOCO images.

Standalone script -- does not import or modify anything under models/, train.py,
eval.py, or predict.py. Run this on any external dataset (e.g. a Roboflow export)
before merging it into the pretraining set, to catch accidental overlap with the
subset-1/4 holdout reserved for final evaluation.

This is a best-effort mitigation, not a guarantee: it catches exact or
near-identical images (same photo, recompressed/cropped/resized). It cannot
catch a different photo of the same scene taken moments apart, which carries
some of the same risk but leaves no matching hash. Flagged pairs should be
eyeballed, not auto-deleted blindly.

Usage:
    uv run scripts/dedup_check.py \
        --holdout dataset/subset-1 dataset/subset-4 \
        --candidates path/to/roboflow_export/images \
        --threshold 5
"""

import argparse
from pathlib import Path

import imagehash
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def hash_directory(directory: Path) -> dict[Path, imagehash.ImageHash]:
    """Compute a perceptual hash for every image file under ``directory``."""
    hashes: dict[Path, imagehash.ImageHash] = {}
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as image:
                hashes[path] = imagehash.phash(image)
        except OSError as error:
            print(f"[skip] could not read {path}: {error}")
    return hashes


def find_matches(
    holdout_hashes: dict[Path, imagehash.ImageHash],
    candidate_hashes: dict[Path, imagehash.ImageHash],
    threshold: int,
) -> list[tuple[Path, Path, int]]:
    """Return (candidate, holdout, hamming_distance) for every pair within threshold."""
    matches = []
    for candidate_path, candidate_hash in candidate_hashes.items():
        for holdout_path, holdout_hash in holdout_hashes.items():
            distance = candidate_hash - holdout_hash
            if distance <= threshold:
                matches.append((candidate_path, holdout_path, distance))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout",
        nargs="+",
        required=True,
        type=Path,
        help="One or more directories containing the reserved subset-1/4 images.",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="Directory of external images being considered for pretraining.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Max Hamming distance between phashes to flag as a match (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flagged.txt"),
        help="Where to write flagged candidate paths (default: flagged.txt).",
    )
    args = parser.parse_args()

    print("Hashing holdout images...")
    holdout_hashes: dict[Path, imagehash.ImageHash] = {}
    for holdout_dir in args.holdout:
        holdout_hashes.update(hash_directory(holdout_dir))
    print(f"  {len(holdout_hashes)} holdout images hashed")

    print("Hashing candidate images...")
    candidate_hashes = hash_directory(args.candidates)
    print(f"  {len(candidate_hashes)} candidate images hashed")

    matches = find_matches(holdout_hashes, candidate_hashes, args.threshold)

    if not matches:
        print("No matches found within threshold.")
        return

    print(f"\n{len(matches)} flagged pairs (candidate -> holdout, distance):")
    with open(args.output, "w") as f:
        for candidate_path, holdout_path, distance in matches:
            line = f"{candidate_path} -> {holdout_path} (distance={distance})"
            print(f"  {line}")
            f.write(f"{candidate_path}\n")
    print(f"\nFlagged candidate paths written to {args.output}")
    print("Review these manually before deciding whether to exclude them.")


if __name__ == "__main__":
    main()
