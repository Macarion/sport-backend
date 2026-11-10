import cv2
import numpy as np

def _hex_to_bgr(hex_color: str, default=(0, 255, 0)):
    if not hex_color:
        return default
    try:
        hex_color = str(hex_color).lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return b, g, r
    except Exception:
        return default


def _point(value):
    return int(round(float(value[0]))), int(round(float(value[1])))


def draw_painting_on_image(img: np.ndarray, painting: list):
    img_copy = img.copy()
    for item in painting or []:
        kind = item.get("kind")
        color = _hex_to_bgr(item.get("color"), default=(0, 255, 0))
        if kind == "bbox":
            x1, y1, x2, y2 = [int(round(float(value))) for value in item.get("xyxy", [0, 0, 0, 0])]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness=2)
            label = str(item.get("target") or "")
            if item.get("confidence") is not None:
                label = f"{label} {float(item['confidence']):.2f}".strip()
            if label:
                cv2.putText(
                    img_copy,
                    label,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        elif kind == "heel" and item.get("point"):
            point = _point(item["point"])
            cv2.circle(img_copy, point, 7, color, thickness=-1)
            cv2.circle(img_copy, point, 12, color, thickness=2)
        elif kind == "text" and item.get("text"):
            position = _point(item.get("position", [24, 42]))
            cv2.putText(
                img_copy,
                str(item["text"]),
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )
        elif kind == "aruco" and item.get("points"):
            points = np.array(item["points"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_copy, [points], isClosed=True, color=color, thickness=2)
            if item.get("anchor"):
                cv2.circle(img_copy, _point(item["anchor"]), 5, (0, 0, 255), thickness=-1)
            if item.get("id") is not None:
                x, y = points.reshape((-1, 2))[0]
                cv2.putText(
                    img_copy,
                    f"ID{item['id']}",
                    (int(x) + 4, max(15, int(y) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return img_copy

