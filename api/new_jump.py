from pathlib import Path
import time
from typing import Dict, Optional

import cv2
import numpy as np
from api.base_sport import BaseSport
from api.jump import FastReadyWebSession

def draw_painting_on_image(img: np.ndarray, painting: list):
    img_copy = img.copy()
    for item in painting:
        if item["kind"] != "bbox":
            continue
        x1, y1, x2, y2 = item["xyxy"]
        # 十六进制颜色转BGR（OpenCV通道BGR）
        hex_color = item["color"].lstrip("#")
        r = int(hex_color[0:2],16)
        g = int(hex_color[2:4],16)
        b = int(hex_color[4:6],16)
        color = (b, g, r)
        # 画矩形框
        cv2.rectangle(img_copy, (x1,y1), (x2,y2), color, thickness=2)
        # 绘制类别+置信度文字
        text = f"{item['target']} {item['confidence']:.2f}"
        cv2.putText(img_copy, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
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

    def start(self) -> dict:
        pass
        # return self.session.to_response(message="测试已开始，请保持 ArUco 标记可见")

    def stop(self):
        if self.session is not None:
            self.session.stop()
            self.session = None

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
                try:
                    painting = draw_painting_on_image(painting, response["painting"])
                except:
                    pass

                del response["painting"]
                return response, painting
            except Exception as exc:
                # print(self.session.fail(f"处理视频帧失败: {exc}", frame_id=frame_id))
                print("处理视频帧失败", exc)
                return None, None
        return None, None

    def jump_status(self, uid, session_id: str) -> dict:
        session = self._get_session_or_none(uid, session_id)
        if session is None:
            return self._missing_session_response(uid, session_id)
        return session.status_response()
