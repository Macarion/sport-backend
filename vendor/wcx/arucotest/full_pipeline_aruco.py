"""Required helpers copied from the existing Web algorithm; source hashes in docs/jump-source-manifest.json.
Unused standalone CLI/model loaders are intentionally omitted.
"""
import cv2
import numpy as np
from PIL import Image

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
        "Selected person box",
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
