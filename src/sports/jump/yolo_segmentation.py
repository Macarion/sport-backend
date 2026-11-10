"""YOLO instance masks adapted to the existing full-resolution measurement contract.

The detection/landing state machine is independent of this module. The existing
tracked person's box selects an instance; it is not a SAM-style model prompt.
"""

import argparse
import time
from dataclasses import asdict, dataclass

import numpy as np

from core.config import VENDOR_DIR
DEFAULT_YOLO_SEG_CHECKPOINT = VENDOR_DIR / "jump" / "yolo11n-seg.pt"


@dataclass(frozen=True)
class YoloSegOptions:
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    match_iou: float = 0.1
    input_mode: str = "full"
    half: bool = False
    max_det: int = 300

    def __post_init__(self):
        if isinstance(self.imgsz, bool) or not isinstance(self.imgsz, int) or self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("yolo-seg-imgsz 必须是 >=32 的 32 倍数，例如 640/960/1280")
        for name in ("conf", "iou", "match_iou"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"yolo-seg-{name.replace('_', '-')} 必须在 [0, 1] 内")
        if self.match_iou == 0:
            raise ValueError("yolo-seg-match-iou 必须大于 0，避免选择无关人物")
        if self.input_mode not in ("full", "roi"):
            raise ValueError("yolo-seg-input 必须是 full 或 roi")
        if not isinstance(self.half, bool):
            raise ValueError("yolo-seg-half 必须是布尔值")
        if isinstance(self.max_det, bool) or not isinstance(self.max_det, int) or self.max_det < 1:
            raise ValueError("yolo-seg-max-det 必须为正整数")

    @classmethod
    def from_args(cls, args):
        return cls(**{name: getattr(args, f"yolo_seg_{name}", field.default)
                      for name, field in cls.__dataclass_fields__.items()})

    def metadata(self):
        return {**asdict(self), "retina_masks": True, "person_class_id": 0}


def _image_size(value):
    size = int(value)
    if size < 32 or size % 32:
        raise argparse.ArgumentTypeError("必须是 >=32 的 32 倍数，例如 640/960/1280")
    return size


def _probability(value):
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("必须是 [0, 1] 内的有限数值")
    return number


def _positive_probability(value):
    number = _probability(value)
    if number == 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def add_yolo_seg_arguments(parser):
    group = parser.add_argument_group("YOLO11n-seg 分割参数（仅 --segmenter yolo11n-seg 生效）")
    group.add_argument("--yolo-seg-checkpoint", default=str(DEFAULT_YOLO_SEG_CHECKPOINT), help="本地分割权重，不自动下载")
    group.add_argument("--yolo-seg-imgsz", type=_image_size, default=640, help="分割网络输入尺度；不改变最终图像坐标")
    group.add_argument("--yolo-seg-conf", type=_probability, default=0.25, help="分割实例的检测置信度阈值，不是 Mask IoU")
    group.add_argument("--yolo-seg-iou", type=_probability, default=0.7, help="分割模型 NMS IoU 阈值")
    group.add_argument("--yolo-seg-match-iou", type=_positive_probability, default=0.1, help="实例框与目标框的最低匹配 IoU")
    group.add_argument("--yolo-seg-input", dest="yolo_seg_input_mode", choices=("full", "roi"), default="full",
                       help="full 整图推理；roi 裁剪外扩人体框后推理并还原全图 Mask")
    group.add_argument("--yolo-seg-half", action="store_true", help="CUDA 上使用 FP16，默认 FP32；CPU 不支持此选项")
    group.add_argument("--yolo-seg-max-det", type=_positive_int, default=300, help="分割推理最多保留的人物实例数")


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


class YoloImageSegmenter:
    def __init__(self, model, torch_module, metadata, options):
        self.backend = "yolo11n-seg"
        self.predictor = model
        self.torch = torch_module
        self.options = options
        self.metadata = {**metadata, "inference": options.metadata()}

    def synchronize(self):
        if self.metadata["device"].startswith("cuda"):
            self.torch.cuda.synchronize(self.metadata["device"])

    def predict(self, landing_frame_bgr, person_box_xyxy, measure=False):
        image = np.asarray(landing_frame_bgr)
        box = np.asarray(person_box_xyxy, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
            raise ValueError("输入图像必须是非空 HxWx3 uint8 BGR")
        if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("人体框必须是有限且宽高为正的 XYXY 坐标")
        height, width = image.shape[:2]
        target = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
        if target[2] <= target[0] or target[3] <= target[1]:
            raise ValueError("人体框与图像没有有效交集")
        x1, y1 = np.floor(target[:2]).astype(int)
        x2, y2 = np.ceil(target[2:]).astype(int)
        roi = self.options.input_mode == "roi"
        offset = np.array([x1, y1, x1, y1]) if roi else np.zeros(4)
        local_target = target - offset
        if measure:
            self.synchronize()
        started = time.perf_counter()
        local_image = np.ascontiguousarray(image[y1:y2, x1:x2] if roi else image)
        results = self.predictor.predict(
            local_image, imgsz=self.options.imgsz, conf=self.options.conf,
            iou=self.options.iou, max_det=self.options.max_det, classes=[0],
            device=self.metadata["device"], half=self.options.half,
            retina_masks=True, verbose=False, save=False,
        )
        if len(results) != 1 or results[0].masks is None or results[0].boxes is None:
            raise ValueError("YOLO11n-seg 未检测到人体 Mask；请检查画面、输入模式或置信度阈值")
        result = results[0]
        boxes, scores, classes = (_numpy(value) for value in (result.boxes.xyxy, result.boxes.conf, result.boxes.cls))
        masks = _numpy(result.masks.data)
        count = len(boxes)
        if (count == 0 or boxes.shape != (count, 4) or scores.shape != (count,) or classes.shape != (count,)
                or masks.shape != (count, *local_image.shape[:2])):
            raise ValueError(f"YOLO11n-seg 输出尺寸不匹配: boxes={boxes.shape}, masks={masks.shape}; 必须使用原尺寸 Mask")
        if not all(np.isfinite(value).all() for value in (boxes, scores, classes, masks)):
            raise ValueError("YOLO11n-seg 输出包含非有限值")
        intersection = np.prod(np.maximum(0, np.minimum(boxes[:, 2:], local_target[2:])
                                         - np.maximum(boxes[:, :2], local_target[:2])), axis=1)
        areas = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
        target_area = np.prod(local_target[2:] - local_target[:2])
        overlaps = intersection / np.maximum(areas + target_area - intersection, 1e-8)
        candidates = [(float(overlaps[i]), float(scores[i]), i) for i in range(count)
                      if classes[i] == 0 and scores[i] >= self.options.conf and overlaps[i] >= self.options.match_iou]
        if not candidates:
            raise ValueError(f"YOLO11n-seg 无匹配测试人的实例：要求目标框 IoU >= {self.options.match_iou:g}；不选择旁人或回退模型")
        overlap, confidence, index = max(candidates)
        selected = (masks[index] > 0).astype(np.uint8)
        if not selected.any():
            raise ValueError("YOLO11n-seg 选中的人体 Mask 为空")
        if roi:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = selected
        else:
            mask = selected
        output = {
            "mask": mask, "best_mask_index": index, "prompt_box": box.tolist(),
            "iou_predictions": [], "model": self.metadata.copy(),
            "quality_semantics": "YOLO instance detection confidence is not mask quality or measured mask IoU",
            "instance_selection": {
                "index": index, "detection_confidence": confidence, "target_box_iou": overlap,
                "instance_box": (boxes[index] + offset).tolist(), "input_mode": self.options.input_mode,
                "crop_bounds": [int(x1), int(y1), int(x2), int(y2)] if roi else [0, 0, width, height],
            },
        }
        if measure:
            self.synchronize()
            output["timing_ms"] = {"total": (time.perf_counter() - started) * 1000}
        return output
