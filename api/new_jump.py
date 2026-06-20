import base64
from pathlib import Path
import time
from typing import Dict, Optional

import cv2
import numpy as np
from api.base_sport import BaseSport
from api.jump import FastReadyWebSession


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


def _decode_data_url_image(value):
    if not isinstance(value, str) or "," not in value:
        return None
    try:
        _, encoded = value.split(",", 1)
        data = base64.b64decode(encoded)
        array = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception:
        return None


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

class JUMP(BaseSport):
    """Library facade for the realtime long-jump measurement backend."""

    def __init__(
        self,
        uid: str = None,
        output_root: Optional[Path | str] = None,
        temp_root: Optional[Path | str] = None,
    ):
        backend_root = Path(__file__).resolve().parent.parent
        self.output_root = Path(output_root) if output_root else backend_root / "outputs"
        self.temp_root = Path(temp_root) if temp_root else backend_root / "temp"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.session = FastReadyWebSession(
            uid=uid,
            output_root=self.output_root,
            temp_root=self.temp_root,
        )
        self._processed_frames = 0
        self._last_log_at = 0.0

    def start(self) -> dict:
        if self.session is None:
            return {}
        return self.session.to_response(message="测试已开始，请保持 ArUco 标记可见")

    def stop(self):
        if self.session is not None:
            response = self.session.stop()
            self.session = None
            return response
        return {}

    def update(
        self,
        frame,
        frame_id: int,
    ) -> dict:
        if self.session is not None:
            try:
                painting = frame.copy()
                response = self.session.process_frame(
                    frame=frame,
                    frame_id=int(frame_id),
                    timestamp_ms=int(time.time() * 1000),
                )
                painting_instructions = response.get("painting") or []
                landing_preview = (response.get("result") or {}).get("landing_frame_preview")
                landing_frame = _decode_data_url_image(landing_preview)
                if landing_frame is not None:
                    painting = landing_frame
                painting = draw_painting_on_image(painting, painting_instructions)
                self._processed_frames += 1
                now = time.time()
                if self._processed_frames == 1 or now - self._last_log_at >= 2.0:
                    self._last_log_at = now
                    print(
                        "[jump] "
                        f"frame={frame_id} state={response.get('state')} "
                        f"message={response.get('message')} "
                        f"score={response.get('score_cm')} "
                        f"painting={len(painting_instructions)}",
                        flush=True,
                    )

                result = dict(response)
                result.pop("painting", None)
                return result, painting
            except Exception as exc:
                # 处理失败时也要把错误回传给前端：返回原始帧 + 错误信息，
                # 而不是 (None, None)。否则 SportManager.process_frames_loop 会丢弃这一帧，
                # 前端只显示本地预览却收不到任何后端响应，看起来像“后端没收到请求”。
                print("处理视频帧失败", exc, flush=True)
                error_result = {
                    "uid": getattr(self.session, "uid", None),
                    "session_id": getattr(self.session, "session_id", None),
                    "frame_id": frame_id,
                    "state": getattr(self.session, "state", "FAILED"),
                    "message": f"处理失败：{exc}",
                    "error": str(exc),
                    "score_cm": None,
                }
                return error_result, frame
        return None, None

    def jump_status(self, uid, session_id: str) -> dict:
        session = self._get_session_or_none(uid, session_id)
        if session is None:
            return self._missing_session_response(uid, session_id)
        return session.status_response()
