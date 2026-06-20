"""
ArUco no-mat jump measurement pipeline.

Step 1 extracts the landing frame with the existing YOLO landing detector.
Step 2 reads a person mask, builds an ArUco plane, finds the heel point, and
measures the heel distance from the takeoff line.

Examples:
    python pipeline_aruco.py --step 1 --video input/test.mp4 --output output/test1
    python pipeline_aruco.py --step 2 --image output/test1/landing_frame.jpg --mask output/test1/person_mask.png --output output/test1
    python pipeline_aruco.py --step 2 --image output/test1/landing_frame.jpg --mask output/test1/person_mask.png --calib-image output/test1/calib.jpg --output output/test1
    python pipeline_aruco.py --step 2 --image output/test1/landing_frame.jpg --mask output/test1/person_mask.png --calib-video input/test.mp4 --output output/test1
"""

import argparse
import importlib.util
from pathlib import Path

import cv2

from aruco_measure import (
    REQUIRED_ARUCO_IDS,
    build_aruco_plane,
    detect_aruco_markers,
    draw_aruco_debug,
    measure_jump_from_files,
)


BASE_DIR = Path(__file__).resolve().parent
JUMP2TEST_DIR = BASE_DIR.parent / "jump2test"


def resolve_existing_path(path_str, base_dir):
    path = Path(path_str)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    base_path = base_dir / path
    if base_path.exists():
        return base_path

    return base_path


def resolve_output_path(path_str, base_dir):
    path = Path(path_str)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.parent.exists():
        return cwd_path

    return base_dir / path


def load_jump_detection_class():
    module_path = JUMP2TEST_DIR / "landing_frame_recognition.py"
    if not module_path.exists():
        raise FileNotFoundError(f"找不到落地帧检测模块: {module_path}")

    spec = importlib.util.spec_from_file_location(
        "jump2test_landing_frame_recognition", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.JumpDetection


def step1_extract_landing_frame(video_path, output_dir):
    print("=" * 80)
    print("Step 1: 提取落地帧")
    print("=" * 80)
    print(f"输入视频: {video_path}")
    print(f"输出目录: {output_dir}")
    print()

    if not video_path.exists():
        print(f"错误: 视频文件不存在 - {video_path}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        JumpDetection = load_jump_detection_class()
        print("正在加载视频并初始化 YOLO 检测器...")
        jump_detector = JumpDetection(str(video_path))
        print(f"[OK] 视频总帧数: {len(jump_detector.frames)}")

        print("正在检测人物进入/离开帧...")
        entry_frame, exit_frame = jump_detector.find_entry_exit_frames()
        print(f"[OK] 人物进入帧: {entry_frame}")
        print(f"[OK] 人物离开帧: {exit_frame}")
        if entry_frame is None or exit_frame is None:
            print("错误: 未能检测到人物进入/离开帧")
            return False

        print("正在检测腾空峰值帧...")
        peak_frame = jump_detector.detect_peak_frame_with_skip(
            start_frame=entry_frame,
            end_frame=exit_frame,
            skip_frames=3,
        )
        if peak_frame is None:
            print("错误: 未能检测到峰值帧")
            return False
        print(f"[OK] 峰值帧: {peak_frame}")

        print("正在检测落地帧...")
        landing_frame_idx = jump_detector.detect_landing_frame_with_window(
            peak_frame=peak_frame,
            end_frame=exit_frame,
            stable_frames=5,
            max_change_threshold=5,
        )
        if landing_frame_idx is None:
            print("错误: 未能检测到落地帧")
            return False
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

        print("=" * 80)
        print("落地帧提取完成")
        print("=" * 80)
        print(f"落地帧: {landing_frame_path}")
        print(f"帧信息: {info_path}")
        print()
        print("下一步: 对落地帧生成人物 mask，然后运行 step 2。")
        print(
            f"示例: python pipeline_aruco.py --step 2 --image \"{landing_frame_path}\" --mask \"<person_mask.png>\" --output \"{output_dir}\""
        )
        return True

    except Exception as exc:
        print(f"错误: {exc}")
        import traceback

        traceback.print_exc()
        return False


def write_result_file(result_path, measurement, image_path, mask_path, calib_image_path):
    heel = measurement.heel_result
    with open(result_path, "w", encoding="utf-8") as file:
        file.write("ArUco 无垫子立定跳远测距结果\n")
        file.write("=" * 40 + "\n")
        file.write(f"落地帧图像: {image_path}\n")
        file.write(f"人物 mask: {mask_path}\n")
        file.write(f"标定图像: {calib_image_path or image_path}\n")
        file.write(f"mask 像素值: {list(measurement.mask_unique_values)}\n")
        file.write(f"人物轮廓数量: {measurement.person_contour_count}\n")
        file.write("\n")
        file.write(f"跳远成绩: {measurement.score_cm:.2f} cm\n")
        file.write(
            "脚后跟图像坐标: "
            f"({heel.image_point[0]:.2f}, {heel.image_point[1]:.2f})\n"
        )
        file.write(
            "脚后跟平面坐标: "
            f"x={heel.world_point[0]:.2f} cm, y={heel.world_point[1]:.2f} cm\n"
        )
        file.write(f"脚后跟候选策略: {heel.candidate_strategy}\n")
        file.write(f"脚底接触带候选点数: {heel.contact_candidate_count}\n")
        file.write("\n")
        file.write(f"ArUco debug 图: {measurement.aruco_debug_path}\n")
        file.write(f"测距可视化图: {measurement.visualization_path}\n")


def find_calibration_frame_in_video(
    video_path,
    output_dir,
    frame_step=30,
    max_samples=300,
):
    """
    Scan a fixed-camera video and save the first sampled frame that detects all
    required ArUco IDs. This avoids requiring a separate calibration photo.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"标定视频不存在: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开标定视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(frame_step))
    max_samples = int(max_samples)

    best_frame = None
    best_frame_idx = None
    best_detections = {}
    best_missing_ids = list(REQUIRED_ARUCO_IDS)

    sampled_count = 0
    frame_idx = 0
    selected_frame = None
    selected_frame_idx = None
    selected_detections = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            detections = detect_aruco_markers(frame)
            missing_ids = [
                marker_id
                for marker_id in REQUIRED_ARUCO_IDS
                if marker_id not in detections
            ]

            if len(detections) > len(best_detections):
                best_frame = frame.copy()
                best_frame_idx = frame_idx
                best_detections = detections
                best_missing_ids = missing_ids

            if not missing_ids:
                selected_frame = frame.copy()
                selected_frame_idx = frame_idx
                selected_detections = detections
                break

            sampled_count += 1
            if max_samples > 0 and sampled_count >= max_samples:
                break

        frame_idx += 1

    cap.release()

    report_path = output_dir / "calib_frame_from_video_report.txt"
    selected_path = output_dir / "calib_frame_from_video.jpg"
    debug_path = output_dir / "calib_frame_from_video_aruco_debug.jpg"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("视频抽帧 ArUco 标定结果\n")
        file.write("=" * 40 + "\n")
        file.write(f"标定视频: {video_path}\n")
        file.write(f"FPS: {fps:.3f}\n")
        file.write(f"总帧数: {total_frames}\n")
        file.write(f"抽帧间隔: {frame_step}\n")
        file.write(f"最大抽样数: {max_samples if max_samples > 0 else '不限制'}\n")
        file.write(f"要求 ID: {list(REQUIRED_ARUCO_IDS)}\n")

        if selected_frame is not None:
            time_sec = selected_frame_idx / fps if fps > 0 else 0.0
            file.write("结论: 成功，已找到可用于标定的帧。\n")
            file.write(f"选中帧: {selected_frame_idx}\n")
            file.write(f"选中时间: {time_sec:.3f} s\n")
            file.write(f"保存图片: {selected_path}\n")
            file.write(f"debug 图片: {debug_path}\n")
        else:
            file.write("结论: 失败，抽样帧中没有任何一帧同时识别到 4 个码。\n")
            file.write(f"最佳帧: {best_frame_idx}\n")
            file.write(f"最佳帧检测到 ID: {sorted(best_detections)}\n")
            file.write(f"最佳帧缺失 ID: {best_missing_ids}\n")
            file.write(f"debug 图片: {debug_path}\n")

    if selected_frame is None:
        if best_frame is not None:
            draw_aruco_debug(
                best_frame,
                best_detections,
                debug_path,
                missing_ids=best_missing_ids,
            )
        raise ValueError(
            "无法从标定视频中找到同时识别 ID 0/1/2/3 的帧；"
            f"详情见 {report_path}"
        )

    cv2.imwrite(str(selected_path), selected_frame)
    build_aruco_plane(selected_frame, debug_path=debug_path)

    print("[OK] 已从视频中抽取 ArUco 标定帧")
    print(f"  标定视频: {video_path}")
    print(f"  选中帧: {selected_frame_idx}")
    print(f"  标定图: {selected_path}")
    print(f"  debug 图: {debug_path}")
    print(f"  报告: {report_path}")
    return selected_path


def step2_measure_distance(image_path, mask_path, output_dir, calib_image_path=None):
    print("=" * 80)
    print("Step 2: ArUco 平面测距")
    print("=" * 80)
    print(f"落地帧图像: {image_path}")
    print(f"人物 mask: {mask_path}")
    print(f"标定图像: {calib_image_path or image_path}")
    print(f"输出目录: {output_dir}")
    print()

    try:
        measurement = measure_jump_from_files(
            image_path=image_path,
            mask_path=mask_path,
            output_dir=output_dir,
            calib_image_path=calib_image_path,
        )

        result_path = output_dir / "result_aruco.txt"
        write_result_file(
            result_path,
            measurement,
            image_path=image_path,
            mask_path=mask_path,
            calib_image_path=calib_image_path,
        )

        print("=" * 80)
        print("ArUco 测距完成")
        print("=" * 80)
        print(f"跳远成绩: {measurement.score_cm:.2f} cm")
        print(
            "脚后跟平面坐标: "
            f"x={measurement.heel_result.world_point[0]:.2f} cm, "
            f"y={measurement.heel_result.world_point[1]:.2f} cm"
        )
        print(f"结果文件: {result_path}")
        print(f"ArUco debug 图: {measurement.aruco_debug_path}")
        print(f"测距可视化图: {measurement.visualization_path}")
        return True

    except Exception as exc:
        print(f"错误: {exc}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="ArUco no-mat jump measurement pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        choices=[1, 2],
        help="执行步骤: 1=提取落地帧, 2=ArUco 测距",
    )
    parser.add_argument("--video", type=str, help="[step1] 输入视频路径")
    parser.add_argument("--image", type=str, help="[step2] 落地帧图像路径")
    parser.add_argument("--mask", type=str, help="[step2] 人物 mask 路径")
    parser.add_argument(
        "--calib-image",
        type=str,
        default=None,
        help="[step2] 可选，同机位空场 ArUco 标定图；默认使用落地帧",
    )
    parser.add_argument(
        "--calib-video",
        type=str,
        default=None,
        help="[step2] 可选，从同机位视频抽取第一帧可识别 4 个 ArUco 的帧作为标定图",
    )
    parser.add_argument(
        "--calib-frame-step",
        type=int,
        default=30,
        help="[step2] --calib-video 抽帧间隔，默认每 30 帧检测一次",
    )
    parser.add_argument(
        "--calib-max-samples",
        type=int,
        default=300,
        help="[step2] --calib-video 最大抽样帧数，0 表示不限制，默认 300",
    )
    parser.add_argument("--output", type=str, required=True, help="输出目录")

    args = parser.parse_args()
    output_dir = resolve_output_path(args.output, BASE_DIR)

    if args.step == 1:
        if not args.video:
            print("错误: step1 需要提供 --video")
            parser.print_help()
            return 1

        video_path = resolve_existing_path(args.video, BASE_DIR)
        success = step1_extract_landing_frame(video_path, output_dir)
        return 0 if success else 1

    if not args.image or not args.mask:
        print("错误: step2 需要同时提供 --image 和 --mask")
        parser.print_help()
        return 1

    image_path = resolve_existing_path(args.image, BASE_DIR)
    mask_path = resolve_existing_path(args.mask, BASE_DIR)
    if args.calib_image and args.calib_video:
        print("错误: --calib-image 和 --calib-video 只能二选一")
        return 1

    if args.calib_video:
        calib_video_path = resolve_existing_path(args.calib_video, BASE_DIR)
        try:
            calib_image_path = find_calibration_frame_in_video(
                video_path=calib_video_path,
                output_dir=output_dir,
                frame_step=args.calib_frame_step,
                max_samples=args.calib_max_samples,
            )
        except Exception as exc:
            print(f"错误: {exc}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        calib_image_path = (
            resolve_existing_path(args.calib_image, BASE_DIR) if args.calib_image else None
        )

    success = step2_measure_distance(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=output_dir,
        calib_image_path=calib_image_path,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
