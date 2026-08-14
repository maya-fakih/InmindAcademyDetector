"""Train-only augmentations for LocoDataset.

Both functions operate on HWC uint8 numpy images and xyxy float boxes,
matching LocoDataset._load_raw's return format, and run before the image is
converted to a CHW float tensor.
"""

import numpy as np
from PIL import Image


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
