#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-dataset}"
download_url="${LOCO_DOWNLOAD_URL:-https://go.mytum.de/239870}"
split_url="https://raw.githubusercontent.com/tum-fml/loco/main/rgb"
annotations=(
    loco-all-v1.json
    loco-sub1-v1-val.json
    loco-sub2-v1-train.json
    loco-sub3-v1-train.json
    loco-sub4-v1-val.json
    loco-sub5-v1-train.json
)

# Checks every image path referenced by the downloaded annotation files actually exists on
# disk (not just that annotation files are present, and not just that *some* image exists).
# Google Drive's FUSE mount can silently drop or truncate individual files out of a few
# thousand written in one go, so "some images exist" is not the same as "all images exist".
check_images_complete() {
    python3 - "$output_dir" "${annotations[@]}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
archive_root = "/dataset"
missing: list[str] = []
checked = 0

for name in sys.argv[2:]:
    if name == "loco-all-v1.json":
        continue  # superset of the per-subset files below; skip to avoid double-checking
    path = output_dir / "rgb" / name
    if not path.is_file():
        sys.exit(1)  # annotation file itself missing -> definitely not ready
    data = json.loads(path.read_text(encoding="utf-8"))
    for image in data["images"]:
        rel = Path(image["path"]).relative_to(archive_root)
        checked += 1
        if not (output_dir / rel).is_file():
            missing.append(str(rel))

print(f"checked {checked} images, {len(missing)} missing", file=sys.stderr)
if missing:
    sample = "\n".join(missing[:20])
    print(f"missing (showing up to 20 of {len(missing)}):\n{sample}", file=sys.stderr)
    sys.exit(1)
PY
}

if check_images_complete; then
    printf 'LOCO is already available at %s\n' "$output_dir"
    exit 0
fi
printf 'LOCO at %s is missing or incomplete; (re)downloading...\n' "$output_dir"

temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT
archive_path="$temporary_dir/loco.zip"
extract_dir="$temporary_dir/extracted"

# ---------------------------------------------------------------------------
# Download images
# ---------------------------------------------------------------------------
printf 'Downloading LOCO images (%.1f MiB)...\n' "$(curl -sI "$download_url" 2>/dev/null | grep -i content-length | tr -d '\r' | awk '{printf "%.1f", $2/1024/1024}')"
curl --fail --location --retry 3 "$download_url" --output "$archive_path"

mkdir -p "$extract_dir"

# Detect archive format and extract accordingly
archive_type="$(file -b --mime-type "$archive_path")"
case "$archive_type" in
    application/zip)
        unzip -q "$archive_path" -d "$extract_dir" ;;
    application/gzip|application/x-tar)
        tar xf "$archive_path" -C "$extract_dir" ;;
    application/x-bzip2)
        tar xjf "$archive_path" -C "$extract_dir" ;;
    *)
        printf 'Unsupported archive format: %s\n' "$archive_type" >&2
        exit 1
esac

# The archive root contains a "dataset/" directory with five subset directories:
#   dataset/subset-{1..5}/
# Extract directly into output_dir so the result is output_dir/subset-{1..5}/.
cp -a "$extract_dir"/dataset/. "$output_dir/"

# ---------------------------------------------------------------------------
# Download COCO annotations from GitHub
# ---------------------------------------------------------------------------
mkdir -p "$output_dir/rgb"
for annotation in "${annotations[@]}"; do
    if [[ ! -f "$output_dir/rgb/$annotation" ]]; then
        curl --fail --location --retry 3 \
            "$split_url/$annotation" \
            --output "$output_dir/rgb/$annotation"
    fi
done

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
for annotation in "${annotations[@]}"; do
    if [[ ! -f "$output_dir/rgb/$annotation" ]]; then
        printf 'LOCO annotation %s could not be downloaded\n' "$annotation" >&2
        exit 1
    fi
done

if ! check_images_complete; then
    printf 'LOCO download finished but some referenced images are still missing from %s ' "$output_dir" >&2
    printf '(see missing list above). This points to a gap in the upstream archive rather than a ' >&2
    printf 'Drive copy glitch, since we just re-extracted it fresh. Consider setting ' >&2
    printf 'LOCO_DOWNLOAD_URL to an alternate mirror.\n' >&2
    exit 1
fi

image_count="$(find "$output_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
printf 'LOCO is ready at %s (%d images, %d annotation files)\n' "$output_dir" "$image_count" "${#annotations[@]}"
