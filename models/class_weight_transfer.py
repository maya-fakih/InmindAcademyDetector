"""Row-level class weight transfer for the YOLO detection head.

``DetectionModel.load()`` (used in yolo_wrapper.py) only copies whole tensors
whose shape matches exactly. When the source and target class counts differ
(e.g. 20-class Roboflow pretrain -> 5-class LOCO), every classification-head
tensor is skipped entirely and randomly reinitialized -- including for the
classes that exist in both.

This module does a finer-grained transfer: for classification-head tensors
(identified by matching ``cv3`` in the key name and a leading dimension equal
to the class count), copy over just the rows for classes present in both the
source and target label sets. Everything else -- backbone, neck, box-regression
head, and rows for target classes with no source match -- is left untouched by
this function; ``DetectionModel.load()`` already handles the shape-matching
transfer for those.

ASSUMPTION FLAGGED FOR VERIFICATION: this identifies classification-head
tensors heuristically (name contains "cv3", leading dim == nc) since we don't
have a live YOLO26 model instance here to introspect the exact module tree.
Before trusting this on a real run, call `list_candidate_tensors` on the
loaded pretrained model and manually confirm the flagged keys are actually
the classification output layers, not e.g. box-regression layers that
happen to share a dimension.
"""

from torch import nn


def list_candidate_tensors(model: nn.Module, num_classes: int) -> list[str]:
    """Return state_dict keys that look like per-class classification tensors.

    Call this on both the pretrained and target models before relying on
    `transfer_matched_class_rows` -- print the results and confirm by eye.
    """
    return [
        key
        for key, tensor in model.state_dict().items()
        if "cv3" in key and tensor.shape[0] == num_classes
    ]


def transfer_matched_class_rows(
    pretrained_model: nn.Module,
    target_model: nn.Module,
    source_class_names: list[str],
    target_class_names: list[str],
) -> list[str]:
    """Copy classification-head rows for classes present in both label sets.

    Args:
        pretrained_model: source model (e.g. loaded from a Roboflow checkpoint).
        target_model: destination model (already built with target nc).
        source_class_names: class names in the order the pretrained head uses.
        target_class_names: class names in the order the target head uses.

    Returns:
        List of class names actually transferred, for logging/verification.
    """
    source_index = {name: i for i, name in enumerate(source_class_names)}
    target_index = {name: i for i, name in enumerate(target_class_names)}
    shared = sorted(set(source_index) & set(target_index))
    if not shared:
        return []

    source_state = pretrained_model.state_dict()
    target_state = target_model.state_dict()
    source_keys = list_candidate_tensors(pretrained_model, len(source_class_names))
    target_keys = list_candidate_tensors(target_model, len(target_class_names))

    if len(source_keys) != len(target_keys):
        raise ValueError(
            f"Found {len(source_keys)} candidate source tensors but "
            f"{len(target_keys)} candidate target tensors -- head structure "
            "may not match between the two models. Aborting transfer rather "
            "than guessing which keys correspond."
        )

    for source_key, target_key in zip(sorted(source_keys), sorted(target_keys), strict=True):
        source_tensor = source_state[source_key]
        target_tensor = target_state[target_key]
        for class_name in shared:
            target_tensor[target_index[class_name]] = source_tensor[source_index[class_name]]

    target_model.load_state_dict(target_state)
    return shared
