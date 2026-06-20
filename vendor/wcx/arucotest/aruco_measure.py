from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


ARUCO_DICTIONARY_NAME = "DICT_6X6_250"
REQUIRED_ARUCO_IDS = (0, 1, 2, 3)

# World plane in centimeters. x is the jump direction, y is left/right.
ARUCO_WORLD_POINTS_CM = {
    0: np.array([0.0, -50.0], dtype=np.float32),
    1: np.array([0.0, 50.0], dtype=np.float32),
    2: np.array([300.0, 50.0], dtype=np.float32),
    3: np.array([300.0, -50.0], dtype=np.float32),
}


@dataclass
class MarkerDetection:
    marker_id: int
    corners: np.ndarray
    anchor_point: np.ndarray


@dataclass
class ArucoPlane:
    image_to_world: np.ndarray
    world_to_image: np.ndarray
    detections: Dict[int, MarkerDetection]
    image_points: np.ndarray
    world_points: np.ndarray


@dataclass
class HeelResult:
    image_point: np.ndarray
    world_point: np.ndarray
    candidate_strategy: str
    contact_candidate_count: int
    contact_candidates_image: np.ndarray
    contact_candidates_world: np.ndarray


@dataclass
class JumpMeasurement:
    score_cm: float
    heel_result: HeelResult
    plane: ArucoPlane
    mask_unique_values: Sequence[int]
    person_contour_count: int
    visualization_path: Path
    aruco_debug_path: Path


def _require_aruco_module():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "当前 OpenCV 缺少 cv2.aruco 模块，请安装 opencv-contrib-python。"
        )

    required_attrs = ["getPredefinedDictionary", "DICT_6X6_250"]
    missing_attrs = [attr for attr in required_attrs if not hasattr(cv2.aruco, attr)]
    if missing_attrs:
        raise RuntimeError(
            "当前 cv2.aruco 缺少必要 API: "
            + ", ".join(missing_attrs)
            + "。请升级或安装 opencv-contrib-python。"
        )


def _get_aruco_dictionary():
    _require_aruco_module()
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


def _create_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return parameters


def detect_aruco_markers(image: np.ndarray) -> Dict[int, MarkerDetection]:
    """Detect ArUco markers and return ID-indexed corners and left-bottom anchors."""
    if image is None:
        raise ValueError("输入图像为空，无法检测 ArUco。")

    dictionary = _get_aruco_dictionary()
    parameters = _create_detector_parameters()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

    detections = {}
    if ids is None:
        return detections

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        normalized_corners = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        marker_id = int(marker_id)
        if marker_id in detections:
            continue

        # OpenCV returns corners clockwise from marker top-left:
        # top-left, top-right, bottom-right, bottom-left.
        anchor_point = normalized_corners[3].copy()
        detections[marker_id] = MarkerDetection(
            marker_id=marker_id,
            corners=normalized_corners,
            anchor_point=anchor_point,
        )

    return detections


def build_aruco_plane(
    image: np.ndarray,
    debug_path: Optional[Path] = None,
) -> ArucoPlane:
    """Build the image->world homography from required ArUco left-bottom anchors."""
    detections = detect_aruco_markers(image)
    missing_ids = [
        marker_id for marker_id in REQUIRED_ARUCO_IDS if marker_id not in detections
    ]

    if missing_ids:
        if debug_path is not None:
            draw_aruco_debug(image, detections, debug_path, missing_ids=missing_ids)
        raise ValueError(
            "缺少必要 ArUco ID: "
            + ", ".join(str(marker_id) for marker_id in missing_ids)
        )

    image_points = np.array(
        [detections[marker_id].anchor_point for marker_id in REQUIRED_ARUCO_IDS],
        dtype=np.float32,
    )
    world_points = np.array(
        [ARUCO_WORLD_POINTS_CM[marker_id] for marker_id in REQUIRED_ARUCO_IDS],
        dtype=np.float32,
    )
    image_to_world = cv2.getPerspectiveTransform(image_points, world_points)
    world_to_image = np.linalg.inv(image_to_world)

    plane = ArucoPlane(
        image_to_world=image_to_world,
        world_to_image=world_to_image,
        detections=detections,
        image_points=image_points,
        world_points=world_points,
    )

    if debug_path is not None:
        draw_aruco_debug(image, detections, debug_path, plane=plane)

    return plane


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def draw_aruco_debug(
    image: np.ndarray,
    detections: Dict[int, MarkerDetection],
    output_path: Path,
    missing_ids: Optional[Sequence[int]] = None,
    plane: Optional[ArucoPlane] = None,
) -> Path:
    debug_image = image.copy()

    for marker_id, detection in sorted(detections.items()):
        corners = detection.corners.astype(np.int32)
        cv2.polylines(debug_image, [corners], True, (0, 255, 0), 2, cv2.LINE_AA)
        for index, corner in enumerate(corners):
            color = (0, 0, 255) if index == 3 else (255, 0, 0)
            cv2.circle(debug_image, tuple(corner), 5, color, -1, cv2.LINE_AA)

        anchor = detection.anchor_point.astype(int)
        cv2.putText(
            debug_image,
            f"ID{marker_id} LB",
            (int(anchor[0]) + 6, int(anchor[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if plane is not None:
        draw_world_reference(debug_image, plane)

    if missing_ids:
        cv2.putText(
            debug_image,
            "Missing IDs: " + ", ".join(str(marker_id) for marker_id in missing_ids),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug_image)
    return output_path


def draw_world_reference(image: np.ndarray, plane: ArucoPlane) -> None:
    world_lines = [
        (np.array([[0.0, -50.0], [0.0, 50.0]], dtype=np.float32), (255, 0, 0), "0 cm"),
        (
            np.array([[300.0, -50.0], [300.0, 50.0]], dtype=np.float32),
            (0, 165, 255),
            "300 cm",
        ),
        (
            np.array([[0.0, -50.0], [300.0, -50.0]], dtype=np.float32),
            (180, 180, 180),
            "left",
        ),
        (
            np.array([[0.0, 50.0], [300.0, 50.0]], dtype=np.float32),
            (180, 180, 180),
            "right",
        ),
        (
            np.array([[0.0, 0.0], [300.0, 0.0]], dtype=np.float32),
            (0, 255, 255),
            "center",
        ),
    ]

    for world_points, color, label in world_lines:
        image_points = transform_points(world_points, plane.world_to_image).astype(int)
        p1 = tuple(image_points[0])
        p2 = tuple(image_points[1])
        cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (p1[0] + 8, p1[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def load_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    return image


def load_person_mask(mask_path: Path, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, Sequence[int]]:
    if not mask_path.exists():
        raise FileNotFoundError(f"人物 mask 不存在: {mask_path}")

    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise ValueError(f"无法读取人物 mask: {mask_path}")

    target_h, target_w = target_shape
    if raw_mask.shape[:2] != (target_h, target_w):
        raw_mask = cv2.resize(
            raw_mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    unique_values = [int(value) for value in np.unique(raw_mask)]
    unique_set = set(unique_values)
    if 1 in unique_set:
        person_mask = (raw_mask == 1).astype(np.uint8)
    elif 255 in unique_set:
        person_mask = (raw_mask == 255).astype(np.uint8)
    else:
        person_mask = (raw_mask > 0).astype(np.uint8)

    return person_mask, unique_values


def extract_person_points(
    person_mask: np.ndarray,
    min_contour_area: float = 50.0,
) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
    contours, _ = cv2.findContours(
        person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= min_contour_area
    ]
    if not valid_contours:
        raise ValueError("未检测到有效人物轮廓，请检查 person mask。")

    person_points = np.vstack(valid_contours).reshape(-1, 2).astype(np.float32)
    return person_points, valid_contours


def _draw_threshold_line(
    image: np.ndarray,
    y_value: float,
    label: str,
    color: Tuple[int, int, int],
    offset: int,
) -> None:
    y_int = int(round(y_value))
    cv2.line(image, (0, y_int), (image.shape[1], y_int), color, 2, cv2.LINE_AA)
    text_y = min(max(20, y_int + offset), image.shape[0] - 10)
    cv2.putText(
        image,
        label,
        (10, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _build_candidate_sets(hull_points: np.ndarray) -> Dict[str, object]:
    person_top = float(np.min(hull_points[:, 1]))
    person_bottom = float(np.max(hull_points[:, 1]))
    person_height = person_bottom - person_top
    if person_height <= 0:
        raise ValueError("人物轮廓高度无效，无法选择脚后跟候选点。")

    lower_third_line = person_bottom - person_height / 3.0
    lower_40_line = person_bottom - 0.4 * person_height
    legacy_percentile_40 = float(np.percentile(hull_points[:, 1], 40))

    return {
        "person_height": person_height,
        "lower_third_line": lower_third_line,
        "lower_40_line": lower_40_line,
        "legacy_percentile_40": legacy_percentile_40,
        "candidate_sets": [
            (
                "lower_third",
                lower_third_line,
                hull_points[hull_points[:, 1] >= lower_third_line],
            ),
            (
                "lower_40_percent",
                lower_40_line,
                hull_points[hull_points[:, 1] >= lower_40_line],
            ),
            (
                "legacy_fallback",
                legacy_percentile_40,
                hull_points[hull_points[:, 1] > legacy_percentile_40],
            ),
        ],
    }


def _select_lower_body_candidates(
    hull_points: np.ndarray,
    image: np.ndarray,
) -> Tuple[np.ndarray, str, Dict[str, object]]:
    candidate_info = _build_candidate_sets(hull_points)

    selected_name = ""
    selected_points = None
    for strategy_name, _, points in candidate_info["candidate_sets"]:
        if len(points) >= 3:
            selected_name = strategy_name
            selected_points = points
            break

    if selected_points is None:
        raise ValueError("脚后跟候选点不足，无法计算脚后跟。")

    _draw_threshold_line(
        image,
        float(candidate_info["lower_third_line"]),
        "Lower 1/3",
        (255, 0, 0),
        -10,
    )
    _draw_threshold_line(
        image,
        float(candidate_info["lower_40_line"]),
        "Lower 40%",
        (0, 165, 255),
        25,
    )
    _draw_threshold_line(
        image,
        float(candidate_info["legacy_percentile_40"]),
        "Legacy 40th pct",
        (255, 255, 0),
        55,
    )

    return selected_points, selected_name, candidate_info


def calculate_heel_point(
    person_points: np.ndarray,
    image: np.ndarray,
    plane: ArucoPlane,
) -> HeelResult:
    """Find heel point from person contour and choose the nearest-to-start candidate."""
    contour = person_points.reshape(-1, 1, 2).astype(np.float32)
    hull = cv2.convexHull(contour)
    hull_points = hull.reshape(-1, 2)
    if len(hull_points) < 3:
        raise ValueError("人物凸包点数不足，无法计算脚后跟。")

    cv2.polylines(image, [hull.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    overlay = image.copy()
    cv2.fillPoly(overlay, [hull.astype(np.int32)], (0, 255, 0))
    image[:] = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)

    for point in hull_points:
        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 0), -1)

    selected_points, strategy_name, candidate_info = _select_lower_body_candidates(
        hull_points, image
    )
    for point in selected_points:
        cv2.circle(image, (int(point[0]), int(point[1])), 4, (255, 0, 0), -1)

    person_height = float(candidate_info["person_height"])
    lowest_y = float(np.max(selected_points[:, 1]))
    contact_band_height = max(10.0, 0.05 * person_height)
    contact_band_y = lowest_y - contact_band_height
    contact_candidates = selected_points[selected_points[:, 1] >= contact_band_y]
    if len(contact_candidates) == 0:
        raise ValueError("脚底接触带为空，无法计算脚后跟。")

    contact_candidates_world = transform_points(
        contact_candidates, plane.image_to_world
    )
    heel_index = int(np.argmin(contact_candidates_world[:, 0]))
    heel_image_point = contact_candidates[heel_index]
    heel_world_point = contact_candidates_world[heel_index]

    _draw_threshold_line(image, contact_band_y, "Heel band", (0, 0, 255), -12)

    for candidate_image, candidate_world in zip(
        contact_candidates, contact_candidates_world
    ):
        cv2.circle(
            image,
            (int(candidate_image[0]), int(candidate_image[1])),
            5,
            (0, 165, 255),
            -1,
        )
        cv2.putText(
            image,
            f"{candidate_world[0]:.0f}",
            (int(candidate_image[0]) + 5, int(candidate_image[1]) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    heel_x, heel_y = int(heel_image_point[0]), int(heel_image_point[1])
    cv2.circle(image, (heel_x, heel_y), 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"Heel {heel_world_point[0]:.1f} cm",
        (heel_x + 8, heel_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return HeelResult(
        image_point=heel_image_point,
        world_point=heel_world_point,
        candidate_strategy=strategy_name,
        contact_candidate_count=len(contact_candidates),
        contact_candidates_image=contact_candidates,
        contact_candidates_world=contact_candidates_world,
    )


def measure_jump_from_files(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    calib_image_path: Optional[Path] = None,
) -> JumpMeasurement:
    output_dir.mkdir(parents=True, exist_ok=True)
    aruco_debug_path = output_dir / "aruco_debug.jpg"
    visualization_path = output_dir / "visualization_aruco.jpg"

    image = load_image(image_path)
    calib_image = load_image(calib_image_path) if calib_image_path else image

    if calib_image.shape[:2] != image.shape[:2]:
        raise ValueError(
            "calib-image 与落地帧尺寸不一致；固定机位标定图必须与落地帧同分辨率。"
        )

    plane = build_aruco_plane(calib_image, aruco_debug_path)
    person_mask, unique_values = load_person_mask(mask_path, image.shape[:2])
    person_points, person_contours = extract_person_points(person_mask)

    visualization = image.copy()
    draw_world_reference(visualization, plane)
    cv2.drawContours(visualization, person_contours, -1, (0, 255, 0), 2)

    heel_result = calculate_heel_point(person_points, visualization, plane)
    cv2.imwrite(str(visualization_path), visualization)

    return JumpMeasurement(
        score_cm=float(heel_result.world_point[0]),
        heel_result=heel_result,
        plane=plane,
        mask_unique_values=unique_values,
        person_contour_count=len(person_contours),
        visualization_path=visualization_path,
        aruco_debug_path=aruco_debug_path,
    )
