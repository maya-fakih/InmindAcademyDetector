#!/usr/bin/env bash
# Download the COCO-pretrained yolov4-tiny backbone (first 29 conv layers) used
# for transfer learning. This is backbone-only on purpose: yolov4-tiny.cfg's
# detection heads here are already resized for LOCO's 5 classes (filters=30,
# not COCO's 255), so the full yolov4-tiny.weights file wouldn't load into the
# head layers cleanly anyway -- Darknet.load_weights() stops once it runs out
# of matching bytes, which is exactly what we want with this file.
set -euo pipefail

output_dir="${1:-weights}"
url="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29"
output_path="$output_dir/yolov4-tiny.conv.29"

mkdir -p "$output_dir"
if [[ -f "$output_path" ]]; then
    printf 'Already downloaded at %s\n' "$output_path"
    exit 0
fi

curl --fail --location --retry 3 "$url" --output "$output_path"
printf 'Saved to %s\n' "$output_path"
