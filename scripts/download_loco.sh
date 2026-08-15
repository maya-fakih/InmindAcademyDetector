#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-dataset}"
mkdir -p "$output_dir"
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

ready=true
for annotation in "${annotations[@]}"; do
    if [[ ! -f "$output_dir/rgb/$annotation" ]]; then
        ready=false
        break
    fi
done
# Images sit at varying depths: subset-1 holds them directly, the others nest them several
# directories down, so the search cannot be depth-limited.
image_file="$(find "$output_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit 2>/dev/null || true)"
if [[ "$ready" == true && -n "$image_file" ]]; then
    printf 'LOCO is already available at %s\n' "$output_dir"
    exit 0
fi

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

image_count="$(find "$output_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
printf 'LOCO is ready at %s (%d images, %d annotation files)\n' "$output_dir" "$image_count" "${#annotations[@]}"
