from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

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


PACKAGE_ROOT = Path(__file__).resolve().parent
API_ROOT = PACKAGE_ROOT.parent
BACKEND_ROOT = API_ROOT.parent
VENDOR_ROOT = BACKEND_ROOT / "vendor"
PROJECT_ROOT = BACKEND_ROOT
WCX_DIR = VENDOR_ROOT / "wcx"
ARUCOTEST_DIR = WCX_DIR / "arucotest"
SAM2_PROJECT_ROOT = WCX_DIR / "sam2"
YOLO_MODEL_PATH = WCX_DIR / "jump2test" / "yolo11n.pt"
SAM2_CHECKPOINT_PATH = WCX_DIR / "sam2" / "checkpoints" / "sam2.1_hiera_small.pt"
SAM2_CONFIG_PATH = WCX_DIR / "sam2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml"

_model_lock = threading.Lock()
_model_bundle = None
_algorithms: Optional[SimpleNamespace] = None


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    wcx_dir: Path
    arucotest_dir: Path


def validate_vendor_assets() -> None:
    required_assets = {
        "实时算法入口 realtime_aruco.py": WCX_DIR / "realtime_aruco.py",
        "fast ready 状态机 realtime_aruco_fast_ready.py": WCX_DIR / "realtime_aruco_fast_ready.py",
        "ArUco 测距模块 aruco_measure.py": ARUCOTEST_DIR / "aruco_measure.py",
        "ArUco pipeline 模块 pipeline_aruco.py": ARUCOTEST_DIR / "pipeline_aruco.py",
        "YOLO 权重 yolo11n.pt": YOLO_MODEL_PATH,
        "SAM2 checkpoint sam2.1_hiera_small.pt": SAM2_CHECKPOINT_PATH,
        "SAM2 config sam2.1_hiera_s.yaml": SAM2_CONFIG_PATH,
    }
    missing = [f"{name}: {path}" for name, path in required_assets.items() if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            "backend/vendor 资源不完整，无法启动 ArUco 跳远测距服务。缺失文件：\n"
            f"{joined}"
        )


def ensure_wcx_import_paths() -> RuntimePaths:
    validate_vendor_assets()
    for path in (WCX_DIR, ARUCOTEST_DIR, SAM2_PROJECT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return RuntimePaths(PROJECT_ROOT, WCX_DIR, ARUCOTEST_DIR)


def get_algorithms() -> SimpleNamespace:
    """Import heavy OpenCV/WCX modules only when a frame is actually processed."""
    global _algorithms
    if _algorithms is not None:
        return _algorithms

    ensure_wcx_import_paths()
    import realtime_aruco as realtime_core
    from aruco_measure import build_aruco_plane, detect_aruco_markers, measure_jump_from_files
    from pipeline_aruco import write_result_file

    _algorithms = SimpleNamespace(
        realtime_core=realtime_core,
        build_aruco_plane=build_aruco_plane,
        detect_aruco_markers=detect_aruco_markers,
        measure_jump_from_files=measure_jump_from_files,
        write_result_file=write_result_file,
    )
    return _algorithms


def get_model_bundle():
    """Load YOLO + SAM2 once and share them across web sessions."""
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    with _model_lock:
        if _model_bundle is not None:
            return _model_bundle

        ensure_wcx_import_paths()
        from realtime_aruco_fast_ready import load_models_without_keypoint

        yolo_path = YOLO_MODEL_PATH
        sam2_checkpoint = SAM2_CHECKPOINT_PATH
        sam2_config = "configs/sam2.1/sam2.1_hiera_s.yaml"
        device = resolve_jump_device()

        _model_bundle = load_models_without_keypoint(
            yolo_model_path=yolo_path,
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            device=device,
        )
        return _model_bundle
