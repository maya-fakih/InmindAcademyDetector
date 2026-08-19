"""Detector factory.

Ported from Amir's exp/03-recipe branch (amiroo-star/inmind-detector) onto
frcnn-amir-recipe so this branch mirrors his actual model choice, not just
his data split. The stock baseline is ``fasterrcnn_mobilenet_v3_large_fpn``;
Amir's recipe uses the 320px variant below instead (see config.yaml
``model.name``). Alternatives are registered alongside it because grading
compares mAP@0.5, total parameters and GFLOPs together, and these occupy
very different points on that surface. Measured on one 1920x1080 image:

    fasterrcnn_mobilenet_v3_large_fpn        18.98M params    42.3 GFLOPs
    fasterrcnn_mobilenet_v3_large_320_fpn    18.98M params     6.8 GFLOPs
    fasterrcnn_resnet50_fpn_v2               43.28M params   745.4 GFLOPs
    ssdlite320_mobilenet_v3_large             2.26M params     0.8 GFLOPs
    slim_mobilenet_fpn (3,6,12,16)/64/512     5.12M params    18.3 GFLOPs
    slim_mobilenet_fpn (12,16)/128/512        7.05M params    19.2 GFLOPs

Every entry keeps torchvision's contract -- ``model(images, targets)`` returns a loss dict
and ``model(images)`` returns detections -- so train, eval and predict work unchanged.
"""

from typing import Any

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

# MobileNetV3-Large stage indices usable as FPN inputs, with their output channels.
# Stage 3 is stride 8, stages 6 and 12 are stride 16, stage 16 is stride 32.
_MOBILENET_STAGE_CHANNELS = {3: 24, 6: 40, 12: 112, 16: 960}


def _faster_rcnn(num_classes: int, weights: Any, builder: Any, **kwargs: Any) -> nn.Module:
    """Build a torchvision Faster R-CNN and resize its box predictor for LOCO."""
    model = builder(weights=weights, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _slim_mobilenet_fpn(
    num_classes: int,
    stages: tuple[int, ...] = (6, 12, 16),
    fpn_channels: int = 64,
    representation_size: int = 512,
    min_size: int = 640,
    max_size: int | None = None,
    pretrained_backbone: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Faster R-CNN on MobileNetV3-Large FPN with configurable width and feature levels."""
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
    features = mobilenet_v3_large(weights=weights).features
    returned_layers = {str(stage): str(index) for index, stage in enumerate(stages)}
    backbone = BackboneWithFPN(
        features,
        returned_layers,
        [_MOBILENET_STAGE_CHANNELS[stage] for stage in stages],
        fpn_channels,
    )
    num_maps = len(stages) + 1
    anchor_generator = AnchorGenerator(
        sizes=tuple([(32, 64, 128, 256, 512)] * num_maps),
        aspect_ratios=tuple([(0.5, 1.0, 2.0)] * num_maps),
    )
    return FasterRCNN(
        backbone,
        num_classes=None,
        min_size=min_size,
        max_size=max_size if max_size is not None else int(min_size * 1.667),
        rpn_anchor_generator=anchor_generator,
        box_head=TwoMLPHead(fpn_channels * 7 * 7, representation_size),
        box_predictor=FastRCNNPredictor(representation_size, num_classes),
        **kwargs,
    )


def _ssdlite(num_classes: int, **kwargs: Any) -> nn.Module:
    """Build SSDLite and resize its classification head."""
    from functools import partial

    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT, **kwargs
    )
    in_channels = [
        module[0][0].in_channels for module in model.head.classification_head.module_list
    ]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes, partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    )
    return model


ARCHITECTURES = {
    "fasterrcnn_mobilenet_v3_large_fpn": lambda n, **kw: _faster_rcnn(
        n,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
        fasterrcnn_mobilenet_v3_large_fpn,
        **kw,
    ),
    "fasterrcnn_mobilenet_v3_large_320_fpn": lambda n, **kw: _faster_rcnn(
        n,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        fasterrcnn_mobilenet_v3_large_320_fpn,
        **kw,
    ),
    "fasterrcnn_resnet50_fpn_v2": lambda n, **kw: _faster_rcnn(
        n, FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, fasterrcnn_resnet50_fpn_v2, **kw
    ),
    "slim_mobilenet_fpn": _slim_mobilenet_fpn,
    "ssdlite320_mobilenet_v3_large": _ssdlite,
}


def create_model(num_classes: int, model_config: dict[str, Any] | None = None) -> nn.Module:
    """Create a pretrained detector; ``num_classes`` includes background.

    ``model_config`` selects the architecture by ``name`` and forwards every other key to
    the builder. Omitting it (``None``, or a config with no ``model:`` section) reproduces
    the original stock baseline exactly, so every other branch in this repo that calls
    ``create_model(num_classes)`` with one argument keeps working unchanged.
    """
    config = dict(model_config or {})
    name = config.pop("name", "fasterrcnn_mobilenet_v3_large_fpn")
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown model '{name}'; choose from {sorted(ARCHITECTURES)}")
    if "stages" in config:
        config["stages"] = tuple(config["stages"])
    return ARCHITECTURES[name](num_classes, **config)
