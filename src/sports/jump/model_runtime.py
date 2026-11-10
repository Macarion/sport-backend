from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from core.config import VENDOR_DIR

logger = logging.getLogger(__name__)


def _setting(name: str, default):
    try:
        from django.conf import settings

        if settings.configured:
            return getattr(settings, name, default)
    except Exception:
        pass
    return default


def _configured_jump_device() -> str:
    value = _setting("JUMP_DEVICE", None)
    if value is None or str(value).strip() == "":
        value = os.environ.get("JUMP_DEVICE", "auto")
    return str(value).strip().lower()


def _cuda_is_available(requested_device: str) -> bool:
    try:
        import torch
    except Exception as exc:
        logger.warning(
            "Unable to import torch while resolving JUMP_DEVICE=%s; falling back to cpu. error=%s",
            requested_device,
            exc,
        )
        return False

    try:
        if not torch.cuda.is_available():
            return False

        if requested_device.startswith("cuda:"):
            _, index_text = requested_device.split(":", 1)
            if index_text.isdigit() and int(index_text) >= torch.cuda.device_count():
                return False

        return True
    except Exception as exc:
        logger.warning(
            "Unable to query CUDA while resolving JUMP_DEVICE=%s; falling back to cpu. error=%s",
            requested_device,
            exc,
        )
        return False


def resolve_jump_device() -> str:
    requested = _configured_jump_device()
    if requested in ("", "auto"):
        return "cuda" if _cuda_is_available("auto") else "cpu"

    if requested == "cpu":
        return "cpu"

    if requested == "cuda" or requested.startswith("cuda:"):
        if _cuda_is_available(requested):
            return requested
        logger.warning("JUMP_DEVICE=%s requested but CUDA is not available; falling back to cpu.", requested)
        return "cpu"

    logger.warning("Unsupported JUMP_DEVICE=%s; using auto device selection.", requested)
    return "cuda" if _cuda_is_available("auto") else "cpu"


# Detection and segmentation are independent, process-local YOLO models.
YOLO_MODEL_PATH = VENDOR_DIR / "jump" / "yolo11n.pt"
YOLO_SEG_MODEL_PATH = VENDOR_DIR / "jump" / "yolo11n-seg.pt"
WCX_DIR = VENDOR_DIR / "wcx"
ARUCOTEST_DIR = WCX_DIR / "arucotest"
_model_lock = threading.Lock()
_model_bundle = None
_algorithms = None


def validate_vendor_assets():
    required = [YOLO_MODEL_PATH, YOLO_SEG_MODEL_PATH, WCX_DIR / "realtime_aruco.py",
                ARUCOTEST_DIR / "aruco_measure.py", ARUCOTEST_DIR / "pipeline_aruco.py",
                ARUCOTEST_DIR / "full_pipeline_aruco.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Jump resources missing: " + "; ".join(missing))


def get_algorithms():
    global _algorithms
    if _algorithms is None:
        for path in (WCX_DIR, ARUCOTEST_DIR):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import realtime_aruco as realtime_core
        from aruco_measure import build_aruco_plane, detect_aruco_markers, measure_jump_from_files
        from pipeline_aruco import write_result_file
        _algorithms = SimpleNamespace(realtime_core=realtime_core,
            build_aruco_plane=build_aruco_plane, detect_aruco_markers=detect_aruco_markers,
            measure_jump_from_files=measure_jump_from_files, write_result_file=write_result_file)
    return _algorithms


def create_yolo_segmenter(checkpoint, device, options=None):
    """YOLO branch migrated from wcx_jump.segmentation (source hash in docs)."""
    import hashlib
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLO segmentation weights missing: {checkpoint}")
    import torch
    import ultralytics
    from ultralytics import YOLO
    from .yolo_segmentation import YoloImageSegmenter, YoloSegOptions
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly select cpu")
    options = options or YoloSegOptions()
    if options.half and requested.type != "cuda":
        raise ValueError("FP16 requires CUDA")
    model = YOLO(str(checkpoint))
    if model.task != "segment" or model.names.get(0) != "person":
        raise ValueError("Expected segmentation weights with COCO person class 0")
    model.to(requested)
    with checkpoint.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    info = {"backend": "yolo11n-seg", "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest, "device": str(requested),
            "precision": "FP16" if options.half else "FP32",
            "source_origin": ultralytics.__file__, "framework_version": ultralytics.__version__,
            "parameter_count": sum(p.numel() for p in model.model.parameters()),
            "weight_bytes": checkpoint.stat().st_size, "task": model.task}
    return YoloImageSegmenter(model, torch, info, options)


def get_model_bundle():
    global _model_bundle
    with _model_lock:
        if _model_bundle is None:
            validate_vendor_assets()
            import torch
            from ultralytics import YOLO
            device = resolve_jump_device()
            detector = YOLO(str(YOLO_MODEL_PATH))
            if detector.task != "detect" or detector.names.get(0) != "person":
                raise ValueError("Expected detection weights with COCO person class 0")
            detector.to(device)
            segmenter = create_yolo_segmenter(YOLO_SEG_MODEL_PATH, device)
            # Finish lazy predictor/CUDA initialization before sending loaded(True),
            # so the first athlete frames do not pay the one-time startup cost.
            import numpy as np
            warmup_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            detector.predict(warmup_frame, verbose=False, device=device)
            segmenter.predictor.predict(warmup_frame, imgsz=640, conf=0.25, iou=0.7,
                                        max_det=300, classes=[0], device=device,
                                        half=False, retina_masks=True, verbose=False)
            segmenter.synchronize()
            _model_bundle = SimpleNamespace(yolo_model=detector, segmenter=segmenter,
                                            keypoint_infer=None, torch_module=torch)
    return _model_bundle


def release_model_bundle():
    global _model_bundle
    with _model_lock:
        _model_bundle = None
