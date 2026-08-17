"""Train-only augmentations for LocoDataset.

Both functions operate on HWC uint8 numpy images and xyxy float boxes,
matching LocoDataset._load_raw's return format, and run before the image is
converted to a CHW float tensor.
"""

import random

import numpy as np
from PIL import Image, ImageEnhance


def random_horizontal_flip(
    image_HWC: np.ndarray, boxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the image left-right and remap box x-coordinates to match."""
    width = image_HWC.shape[1]
    flipped_image = image_HWC[:, ::-1, :].copy()
    flipped_boxes = boxes.copy()
    if boxes.shape[0] > 0:
        flipped_boxes[:, 0] = width - boxes[:, 2]
        flipped_boxes[:, 2] = width - boxes[:, 0]
    return flipped_image, flipped_boxes


def random_color_jitter(
    image_HWC: np.ndarray,
    brightness_range: tuple[float, float] = (0.8, 1.2),
    contrast_range: tuple[float, float] = (0.8, 1.2),
    saturation_range: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """Randomly perturb brightness/contrast/saturation. Boxes are untouched --
    this is a pure appearance change, useful for LOCO's varied warehouse
    lighting (different sites, times of day, artificial vs. natural light).
    """
    image = Image.fromarray(image_HWC)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(*brightness_range))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(*contrast_range))
    image = ImageEnhance.Color(image).enhance(random.uniform(*saturation_range))
    return np.asarray(image, dtype=np.uint8)


def random_scale_jitter(
    image_HWC: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scale_range: tuple[float, float] = (0.75, 1.3),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zoom the image in/out around a random center, keeping the output the
    same H/W (crop if zoomed in, pad if zoomed out) so downstream letterboxing
    is unaffected. Boxes falling entirely outside the kept region are dropped
    (with their labels) rather than left as degenerate zero-area boxes.

    Motivation: LOCO objects appear at a wide range of distances/scales across
    its warehouses; the model otherwise only ever sees each object's one
    as-photographed scale.
    """
    height, width = image_HWC.shape[:2]
    scale = random.uniform(*scale_range)
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    resized = np.asarray(Image.fromarray(image_HWC).resize((new_width, new_height), Image.BILINEAR))

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Where the resized image's top-left lands on the fixed-size canvas -- negative
    # when zoomed in (we're cropping), positive when zoomed out (we're padding).
    offset_y = random.randint(min(0, height - new_height), max(0, height - new_height))
    offset_x = random.randint(min(0, width - new_width), max(0, width - new_width))

    src_y0, src_x0 = max(0, -offset_y), max(0, -offset_x)
    dst_y0, dst_x0 = max(0, offset_y), max(0, offset_x)
    copy_h = min(new_height - src_y0, height - dst_y0)
    copy_w = min(new_width - src_x0, width - dst_x0)
    canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = resized[
        src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
    ]

    if boxes.shape[0] == 0:
        return canvas, boxes, labels

    jittered = boxes * scale
    jittered[:, [0, 2]] += offset_x
    jittered[:, [1, 3]] += offset_y
    jittered[:, [0, 2]] = jittered[:, [0, 2]].clip(0, width)
    jittered[:, [1, 3]] = jittered[:, [1, 3]].clip(0, height)

    keep = (jittered[:, 2] - jittered[:, 0] > 1) & (jittered[:, 3] - jittered[:, 1] > 1)
    return canvas, jittered[keep], labels[keep]


def random_background_swap(
    image_HWC: np.ndarray,
    boxes: np.ndarray,
    donor_image_HWC: np.ndarray,
    donor_boxes: np.ndarray,
) -> np.ndarray:
    """Replace everything outside ``boxes`` with ``donor_image_HWC``'s background.

    Keeps the labeled objects (exact pixels, so boxes stay valid) from
    ``image_HWC``, but swaps the surrounding warehouse scene for a different
    warehouse's background. Motivation: LOCO's own train/eval split is
    cross-warehouse by design (Mayershofer et al. 2020) specifically to
    resist background-context shortcuts; this augmentation pushes the same
    idea further during training. See Handoff.md.

    Known limitation: ``donor_boxes`` is accepted but not used to mask out
    the donor's own objects, so a donor object can appear in the composited
    background. Stated here rather than silently ignored -- worth fixing if
    this augmentation turns out to help but background swap noise (a donor
    forklift bleeding into a pallet image) turns out to hurt more than the
    intended effect helps.
    """
    height, width = image_HWC.shape[:2]
    if donor_image_HWC.shape[:2] != (height, width):
        donor_resized = np.asarray(
            Image.fromarray(donor_image_HWC).resize((width, height), Image.BILINEAR)
        )
    else:
        donor_resized = donor_image_HWC

    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes.astype(int):
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width), min(y2, height)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True

    composited = np.where(mask[:, :, None], image_HWC, donor_resized)
    return composited.astype(np.uint8)
