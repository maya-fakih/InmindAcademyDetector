"""Flag candidate pretraining images that perceptually match held-out LOCO images.

Standalone script -- does not import or modify anything under models/, train.py,
eval.py, or predict.py. Run this on any external dataset (e.g. a Roboflow export)
before merging it into the pretraining set, to catch accidental overlap with the
subset-1/4 holdout reserved for final evaluation.

This is a best-effort mitigation, not a guarantee: it catches exact or
near-identical images (same photo, recompressed/cropped/resized), and now also
rotated (90/180/270) or horizontally-flipped re-exports of a leaked image, but
only for candidates that are already "borderline" close to a holdout image
under a direct perceptual-hash comparison -- checking every candidate's
rotations against every holdout image regardless of how unrelated they are
would be needlessly expensive for negligible benefit. It cannot catch a
different photo of the same scene taken moments apart, which carries some of
the same risk but leaves no matching hash. Flagged pairs should be eyeballed,
not auto-deleted blindly.

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


def coarse_signature(image: Image.Image) -> list[float]:
    """A cheap, roughly rotation/flip-invariant signature (normalized color
    histogram) used only to decide whether a candidate is "sketchy" enough to
    warrant the expensive rotated-phash check below. Rotating or flipping an
    image barely changes its overall color distribution, so this catches
    rotated/flipped duplicates that a direct phash comparison would miss
    entirely (phash itself is NOT rotation-invariant -- a 90-degree rotation of
    the exact same image can have a very different hash).
    """
    histogram = image.convert("RGB").resize((64, 64)).histogram()
    total = sum(histogram) or 1
    return [count / total for count in histogram]


def histogram_distance(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b, strict=True))


def find_matches(
    holdout_hashes: dict[Path, imagehash.ImageHash],
    candidate_hashes: dict[Path, imagehash.ImageHash],
    holdout_signatures: dict[Path, list[float]],
    candidate_signatures: dict[Path, list[float]],
    threshold: int,
    histogram_threshold: float = 0.4,
) -> list[tuple[Path, Path, int, str]]:
    """Return (candidate, holdout, hamming_distance, method) for every match.

    Two passes:
    1. Direct phash comparison at `threshold` (catches identical/near-identical
       images with the same orientation).
    2. For candidates whose coarse color-histogram signature is close to some
       holdout image (below `histogram_threshold`) but weren't already caught
       by pass 1, also check the candidate rotated 90/180/270 degrees and
       horizontally flipped against every holdout hash. This only runs for
       that "sketchy" subset, not every candidate, since re-hashing 4 extra
       orientations for every candidate against every holdout image would be
       needlessly expensive when the vast majority of pairs are unrelated.
    """
    matches: list[tuple[Path, Path, int, str]] = []
    sketchy_candidates: set[Path] = set()

    for candidate_path, candidate_hash in candidate_hashes.items():
        matched_directly = False
        for holdout_path, holdout_hash in holdout_hashes.items():
            distance = candidate_hash - holdout_hash
            if distance <= threshold:
                matches.append((candidate_path, holdout_path, distance, "direct"))
                matched_directly = True
        if matched_directly:
            continue
        candidate_sig = candidate_signatures[candidate_path]
        if any(
            histogram_distance(candidate_sig, holdout_sig) <= histogram_threshold
            for holdout_sig in holdout_signatures.values()
        ):
            sketchy_candidates.add(candidate_path)

    if sketchy_candidates:
        print(
            f"  {len(sketchy_candidates)} candidates flagged as color-similar -- "
            f"re-checking against rotations/flip..."
        )
    for candidate_path in sketchy_candidates:
        try:
            with Image.open(candidate_path) as image:
                variants = {
                    "rot90": image.rotate(90, expand=True),
                    "rot180": image.rotate(180, expand=True),
                    "rot270": image.rotate(270, expand=True),
                    "hflip": image.transpose(Image.FLIP_LEFT_RIGHT),
                }
                variant_hashes = {
                    name: imagehash.phash(variant) for name, variant in variants.items()
                }
        except OSError as error:
            print(f"[skip] could not re-check {candidate_path}: {error}")
            continue

        for holdout_path, holdout_hash in holdout_hashes.items():
            for method, variant_hash in variant_hashes.items():
                distance = variant_hash - holdout_hash
                if distance <= threshold:
                    matches.append((candidate_path, holdout_path, distance, method))

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
    holdout_signatures: dict[Path, list[float]] = {}
    for holdout_dir in args.holdout:
        for path in sorted(holdout_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    holdout_hashes[path] = imagehash.phash(image)
                    holdout_signatures[path] = coarse_signature(image)
            except OSError as error:
                print(f"[skip] could not read {path}: {error}")
    print(f"  {len(holdout_hashes)} holdout images hashed")

    print("Hashing candidate images...")
    candidate_hashes: dict[Path, imagehash.ImageHash] = {}
    candidate_signatures: dict[Path, list[float]] = {}
    for path in sorted(args.candidates.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as image:
                candidate_hashes[path] = imagehash.phash(image)
                candidate_signatures[path] = coarse_signature(image)
        except OSError as error:
            print(f"[skip] could not read {path}: {error}")
    print(f"  {len(candidate_hashes)} candidate images hashed")

    matches = find_matches(
        holdout_hashes, candidate_hashes, holdout_signatures, candidate_signatures, args.threshold
    )

    if not matches:
        print("No matches found within threshold.")
        return

    print(f"\n{len(matches)} flagged pairs (candidate -> holdout, distance, method):")
    written = set()
    with open(args.output, "w") as f:
        for candidate_path, holdout_path, distance, method in matches:
            line = f"{candidate_path} -> {holdout_path} (distance={distance}, method={method})"
            print(f"  {line}")
            if candidate_path not in written:
                f.write(f"{candidate_path}\n")
                written.add(candidate_path)
    print(f"\nFlagged candidate paths written to {args.output}")
    print("Review these manually before deciding whether to exclude them.")


if __name__ == "__main__":
    main()
