"""
Full fixed-camera ArUco jump measurement pipeline.

Input one video and run the complete flow:
1. Extract an ArUco calibration frame from the same fixed-camera video.
2. Detect the landing frame with the existing YOLO landing-frame logic.
3. Detect a person bbox on the landing frame and use it as a SAM2 image prompt.
4. Save a person-only mask for the landing frame.
5. Measure heel distance with the ArUco plane.

Example:
    python wcx/arucotest/full_pipeline_aruco.py \
        --video test5.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from aruco_measure import measure_jump_from_files
from pipeline_aruco import (
    find_calibration_frame_in_video,
    load_jump_detection_class,
    resolve_existing_path,
    write_result_file,
)


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
SAM2_PROJECT_ROOT = BASE_DIR.parent / "sam2"
DEFAULT_SAM2_CHECKPOINT = (
    SAM2_PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_DEVICE = "cuda"


def configure_console_encoding():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


configure_console_encoding()


def normalize_sam2_config(config_value):
    """Normalize a SAM2 config path to the Hydra config name expected by SAM2."""
    config_str = str(config_value).replace("\\", "/")
    if config_str.startswith("sam2/"):
        config_str = config_str[len("sam2/") :]

    package_root = SAM2_PROJECT_ROOT / "sam2"
    direct_candidate = package_root / config_str
    if direct_candidate.exists():
        return config_str

    absolute_candidate = Path(config_value)
    if absolute_candidate.is_absolute() and absolute_candidate.exists():
        try:
            return absolute_candidate.relative_to(package_root).as_posix()
        except ValueError:
            return config_str

    cwd_candidate = Path.cwd() / config_value
    if cwd_candidate.exists():
        try:
            return cwd_candidate.relative_to(package_root).as_posix()
        except ValueError:
            return config_str

    project_candidate = SAM2_PROJECT_ROOT / config_value
    if project_candidate.exists():
        try:
            return project_candidate.relative_to(package_root).as_posix()
        except ValueError:
            return config_str

    return config_str


def resolve_video_path(video_value):
    """Resolve video name or path, preferring wcx/arucotest/input for bare names."""
    path = Path(video_value)
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    base_candidate = BASE_DIR / path
    if base_candidate.exists():
        return base_candidate

    if len(path.parts) == 1:
        return INPUT_DIR / path.name

    return base_candidate


def resolve_full_output_dir(output_value, video_path):
    """Use explicit output if provided; otherwise output/<video stem>."""
    if output_value:
        path = Path(output_value)
        if path.is_absolute():
            return path

        cwd_path = Path.cwd() / path
        if cwd_path.parent.exists():
            return cwd_path

        return BASE_DIR / path

    return OUTPUT_DIR / video_path.stem


def ensure_sam2_import_path():
    sam2_path = str(SAM2_PROJECT_ROOT)
    if sam2_path not in sys.path:
        sys.path.insert(0, sam2_path)


def parse_person_box(box_text):
    if not box_text:
        return None

    values = [int(float(value.strip())) for value in box_text.split(",")]
    if len(values) != 4:
        raise ValueError("--person-box 格式应为 x1,y1,x2,y2")

    x1, y1, x2, y2 = values
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y2 = int(max(0, min(height - 1, round(y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"人物框无效: {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]


def expand_box(box, image_shape, expand_ratio):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * expand_ratio
    pad_y = box_h * expand_ratio
    return clamp_box(
        [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y],
        width=width,
        height=height,
    )


def detect_largest_person_box(yolo_model, image, expand_ratio=0.08):
    results = yolo_model(image)
    candidates = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id != 0:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            if area <= 0:
                continue

            candidates.append(
                {
                    "box": [x1, y1, x2, y2],
                    "area": area,
                    "confidence": conf,
                }
            )

    if not candidates:
        raise ValueError(
            "YOLO 未在落地帧中检测到人物；可改用 --prompt-mode manual 或 "
            '--prompt-mode box --person-box "x1,y1,x2,y2"。'
        )

    candidates.sort(key=lambda item: item["area"], reverse=True)
    selected = candidates[0]
    original_box = clamp_box(
        selected["box"], width=image.shape[1], height=image.shape[0]
    )
    expanded_box = expand_box(original_box, image.shape, expand_ratio)
    return {
        "original_box": original_box,
        "expanded_box": expanded_box,
        "confidence": selected["confidence"],
        "detected_person_count": len(candidates),
    }


def select_person_box_manual(image):
    window_name = "Select person box - SPACE/ENTER confirms"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(
        window_name,
        image,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError("未框选人物。")
    return clamp_box([x, y, x + w, y + h], width=image.shape[1], height=image.shape[0])


def draw_person_box_debug(image, bbox_info, output_path):
    debug = image.copy()
    if bbox_info.get("original_box"):
        x1, y1, x2, y2 = bbox_info["original_box"]
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            debug,
            "YOLO box",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    x1, y1, x2, y2 = bbox_info["box"]
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        debug,
        "SAM2 prompt box",
        (x1, min(debug.shape[0] - 10, y2 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), debug)


def create_colored_person_mask(mask):
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored[mask > 0] = (0, 255, 0)
    return colored


def save_segmentation_outputs(image, mask, bbox_info, output_dir):
    mask_path = output_dir / "person_mask.png"
    colored_path = output_dir / "person_mask_colored.png"
    overlay_path = output_dir / "person_segmentation_visualization.jpg"
    bbox_debug_path = output_dir / "person_bbox_debug.jpg"

    Image.fromarray(mask.astype(np.uint8)).save(mask_path)

    colored_rgb = create_colored_person_mask(mask)
    colored_bgr = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(colored_path), colored_bgr)

    overlay = image.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (
        0.45 * np.array([0, 255, 0]) + 0.55 * overlay[mask_bool]
    ).astype(np.uint8)
    cv2.imwrite(str(overlay_path), overlay)

    draw_person_box_debug(image, bbox_info, bbox_debug_path)
    return {
        "mask_path": mask_path,
        "colored_path": colored_path,
        "overlay_path": overlay_path,
        "bbox_debug_path": bbox_debug_path,
    }


def segment_landing_frame_with_sam2(
    landing_frame,
    person_box,
    output_dir,
    sam2_checkpoint,
    sam2_config,
    device,
):
    ensure_sam2_import_path()
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_config = normalize_sam2_config(sam2_config)
    print("正在加载 SAM2 image predictor...")
    sam_model = build_sam2(
        config_file=sam2_config,
        ckpt_path=str(sam2_checkpoint),
        device=device,
    )
    predictor = SAM2ImagePredictor(sam_model)

    image_rgb = cv2.cvtColor(landing_frame, cv2.COLOR_BGR2RGB)
    prompt_box = np.array(person_box, dtype=np.float32)

    with torch.inference_mode():
        predictor.set_image(image_rgb)
        masks, iou_predictions, _ = predictor.predict(
            box=prompt_box,
            multimask_output=True,
        )

    best_index = int(np.argmax(iou_predictions))
    mask = (masks[best_index] > 0).astype(np.uint8)
    print(
        f"[OK] SAM2 人像分割完成: best_mask={best_index}, "
        f"iou={float(iou_predictions[best_index]):.4f}"
    )
    return {
        "mask": mask,
        "best_mask_index": best_index,
        "iou_predictions": [float(value) for value in iou_predictions],
    }


def extract_landing_frame(video_path, output_dir):
    JumpDetection = load_jump_detection_class()
    print("正在加载视频并初始化 YOLO 检测器...")
    jump_detector = JumpDetection(str(video_path))
    print(f"[OK] 视频总帧数: {len(jump_detector.frames)}")

    print("正在检测人物进入/离开帧...")
    entry_frame, exit_frame = jump_detector.find_entry_exit_frames()
    if entry_frame is None or exit_frame is None:
        raise ValueError("未能检测到人物进入/离开帧。")
    print(f"[OK] 人物进入帧: {entry_frame}")
    print(f"[OK] 人物离开帧: {exit_frame}")

    print("正在检测腾空峰值帧...")
    peak_frame = jump_detector.detect_peak_frame_with_skip(
        start_frame=entry_frame,
        end_frame=exit_frame,
        skip_frames=3,
    )
    if peak_frame is None:
        raise ValueError("未能检测到峰值帧。")
    print(f"[OK] 峰值帧: {peak_frame}")

    print("正在检测落地帧...")
    landing_frame_idx = jump_detector.detect_landing_frame_with_window(
        peak_frame=peak_frame,
        end_frame=exit_frame,
        stable_frames=5,
        max_change_threshold=5,
    )
    if landing_frame_idx is None:
        raise ValueError("未能检测到落地帧。")
    print(f"[OK] 落地帧: {landing_frame_idx}")

    landing_frame = jump_detector.frames[landing_frame_idx]
    landing_frame_path = output_dir / "landing_frame.jpg"
    cv2.imwrite(str(landing_frame_path), landing_frame)

    info_path = output_dir / "frame_info.txt"
    with open(info_path, "w", encoding="utf-8") as file:
        file.write(f"视频路径: {video_path}\n")
        file.write(f"总帧数: {len(jump_detector.frames)}\n")
        file.write(f"进入帧: {entry_frame}\n")
        file.write(f"峰值帧: {peak_frame}\n")
        file.write(f"落地帧: {landing_frame_idx}\n")
        file.write(f"离开帧: {exit_frame}\n")
        file.write(f"图像尺寸: {landing_frame.shape[1]}x{landing_frame.shape[0]}\n")

    return {
        "jump_detector": jump_detector,
        "landing_frame": landing_frame,
        "landing_frame_path": landing_frame_path,
        "frame_info_path": info_path,
        "entry_frame": entry_frame,
        "exit_frame": exit_frame,
        "peak_frame": peak_frame,
        "landing_frame_idx": landing_frame_idx,
        "total_frames": len(jump_detector.frames),
    }


def get_person_box(prompt_mode, landing_frame, jump_detector, person_box_text, expand_ratio):
    if prompt_mode == "box":
        box = parse_person_box(person_box_text)
        if box is None:
            raise ValueError('--prompt-mode box 需要提供 --person-box "x1,y1,x2,y2"')
        box = clamp_box(box, width=landing_frame.shape[1], height=landing_frame.shape[0])
        return {
            "box": box,
            "original_box": None,
            "confidence": None,
            "detected_person_count": None,
            "source": "box",
        }

    if prompt_mode == "manual":
        box = select_person_box_manual(landing_frame)
        return {
            "box": box,
            "original_box": None,
            "confidence": None,
            "detected_person_count": None,
            "source": "manual",
        }

    yolo_info = detect_largest_person_box(
        jump_detector.model,
        landing_frame,
        expand_ratio=expand_ratio,
    )
    return {
        "box": yolo_info["expanded_box"],
        "original_box": yolo_info["original_box"],
        "confidence": yolo_info["confidence"],
        "detected_person_count": yolo_info["detected_person_count"],
        "source": "auto_yolo",
    }


def write_pipeline_json(
    output_path,
    video_path,
    calib_frame_path,
    landing_info,
    bbox_info,
    segmentation_info,
    measurement,
):
    heel = measurement.heel_result
    payload = {
        "video": str(video_path),
        "calib_frame": str(calib_frame_path),
        "landing_frame": str(landing_info["landing_frame_path"]),
        "frame_info": {
            "total_frames": landing_info["total_frames"],
            "entry_frame": landing_info["entry_frame"],
            "peak_frame": landing_info["peak_frame"],
            "landing_frame": landing_info["landing_frame_idx"],
            "exit_frame": landing_info["exit_frame"],
        },
        "person_box": bbox_info,
        "segmentation": {
            "best_mask_index": segmentation_info["best_mask_index"],
            "iou_predictions": segmentation_info["iou_predictions"],
        },
        "measurement": {
            "score_cm": measurement.score_cm,
            "heel_image_point": [
                float(heel.image_point[0]),
                float(heel.image_point[1]),
            ],
            "heel_world_point": [
                float(heel.world_point[0]),
                float(heel.world_point[1]),
            ],
            "candidate_strategy": heel.candidate_strategy,
            "contact_candidate_count": heel.contact_candidate_count,
        },
        "outputs": {
            "result": str(output_path.parent / "result_aruco.txt"),
            "visualization_aruco": str(measurement.visualization_path),
            "aruco_debug": str(measurement.aruco_debug_path),
            "person_mask": str(output_path.parent / "person_mask.png"),
            "person_segmentation_visualization": str(
                output_path.parent / "person_segmentation_visualization.jpg"
            ),
        },
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def run_full_pipeline(args):
    video_path = resolve_video_path(args.video)
    output_dir = resolve_full_output_dir(args.output, video_path)
    sam2_checkpoint = resolve_existing_path(args.sam2_checkpoint, BASE_DIR)
    sam2_config = normalize_sam2_config(args.sam2_config)

    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if not sam2_checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint 不存在: {sam2_checkpoint}")

    sam2_config_path = SAM2_PROJECT_ROOT / "sam2" / sam2_config
    if not sam2_config_path.exists():
        raise FileNotFoundError(f"SAM2 config 不存在: {sam2_config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 输入视频: {video_path}")
    print(f"[OK] 输出目录: {output_dir}")

    print("=" * 80)
    print("Step 1: 从视频抽取 ArUco 标定帧")
    print("=" * 80)
    calib_frame_path = find_calibration_frame_in_video(
        video_path=video_path,
        output_dir=output_dir,
        frame_step=args.calib_frame_step,
        max_samples=args.calib_max_samples,
    )

    print("=" * 80)
    print("Step 2: 检测落地帧")
    print("=" * 80)
    landing_info = extract_landing_frame(video_path, output_dir)
    landing_frame = landing_info["landing_frame"]

    print("=" * 80)
    print("Step 3: 自动获取人物框")
    print("=" * 80)
    bbox_info = get_person_box(
        prompt_mode=args.prompt_mode,
        landing_frame=landing_frame,
        jump_detector=landing_info["jump_detector"],
        person_box_text=args.person_box,
        expand_ratio=args.bbox_expand_ratio,
    )
    draw_person_box_debug(
        landing_frame,
        bbox_info,
        output_dir / "person_bbox_debug.jpg",
    )
    print(f"[OK] 人物框来源: {bbox_info['source']}")
    print(f"[OK] SAM2 prompt box: {bbox_info['box']}")

    print("=" * 80)
    print("Step 4: SAM2 单图人像分割")
    print("=" * 80)
    segmentation_info = segment_landing_frame_with_sam2(
        landing_frame=landing_frame,
        person_box=bbox_info["box"],
        output_dir=output_dir,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        device=args.device,
    )
    output_paths = save_segmentation_outputs(
        landing_frame,
        segmentation_info["mask"],
        bbox_info,
        output_dir,
    )
    print(f"[OK] 人物 mask: {output_paths['mask_path']}")
    print(f"[OK] 分割可视化: {output_paths['overlay_path']}")

    print("=" * 80)
    print("Step 5: 脚后跟点 + ArUco 测距")
    print("=" * 80)
    measurement = measure_jump_from_files(
        image_path=landing_info["landing_frame_path"],
        mask_path=output_paths["mask_path"],
        output_dir=output_dir,
        calib_image_path=calib_frame_path,
    )
    result_path = output_dir / "result_aruco.txt"
    write_result_file(
        result_path,
        measurement,
        image_path=landing_info["landing_frame_path"],
        mask_path=output_paths["mask_path"],
        calib_image_path=calib_frame_path,
    )

    pipeline_json_path = output_dir / "pipeline_result.json"
    write_pipeline_json(
        output_path=pipeline_json_path,
        video_path=video_path,
        calib_frame_path=calib_frame_path,
        landing_info=landing_info,
        bbox_info=bbox_info,
        segmentation_info=segmentation_info,
        measurement=measurement,
    )

    print("=" * 80)
    print("完整流程完成")
    print("=" * 80)
    print(f"跳远成绩: {measurement.score_cm:.2f} cm")
    print(f"结果文件: {result_path}")
    print(f"完整 JSON: {pipeline_json_path}")
    print(f"测距可视化: {measurement.visualization_path}")
    print("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Full fixed-camera ArUco jump measurement pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="输入视频名或路径；只写文件名时默认从 wcx/arucotest/input/ 读取",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录；不填时默认 wcx/arucotest/output/<视频文件名>",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default=str(DEFAULT_SAM2_CHECKPOINT),
        help="SAM2 checkpoint 路径",
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default=DEFAULT_SAM2_CONFIG,
        help="SAM2 config 名称或路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="运行设备，默认 cuda；需要 CPU 时传 cpu",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["auto", "box", "manual"],
        default="auto",
        help="人物分割 prompt 来源: auto=YOLO自动框人, box=指定框, manual=弹窗框选",
    )
    parser.add_argument(
        "--person-box",
        type=str,
        default=None,
        help='--prompt-mode box 时使用，格式 "x1,y1,x2,y2"',
    )
    parser.add_argument(
        "--bbox-expand-ratio",
        type=float,
        default=0.08,
        help="YOLO 人框外扩比例，默认 0.08",
    )
    parser.add_argument(
        "--calib-frame-step",
        type=int,
        default=30,
        help="视频抽帧找 ArUco 标定帧的间隔，默认 30",
    )
    parser.add_argument(
        "--calib-max-samples",
        type=int,
        default=300,
        help="视频抽帧最大样本数，0 表示不限制，默认 300",
    )

    args = parser.parse_args()

    try:
        return 0 if run_full_pipeline(args) else 2
    except Exception as exc:
        print(f"错误: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
