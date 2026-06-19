from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, Optional

from .session import FastReadyWebSession


class JUMP:
    """Library facade for the realtime long-jump measurement backend."""

    def __init__(
        self,
        output_root: Optional[Path | str] = None,
        temp_root: Optional[Path | str] = None,
    ):
        backend_root = Path(__file__).resolve().parent.parent
        self.output_root = Path(output_root) if output_root else backend_root / "outputs"
        self.temp_root = Path(temp_root) if temp_root else backend_root / "temp"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, FastReadyWebSession] = {}
        self._store_lock = threading.Lock()

    def jump_start(self, uid) -> dict:
        uid = str(uid)
        session = FastReadyWebSession(
            uid=uid,
            output_root=self.output_root,
            temp_root=self.temp_root,
        )
        with self._store_lock:
            self._sessions[uid] = session
        return session.to_response(message="测试已开始，请保持 ArUco 标记可见")

    def jump_process_frame(
        self,
        uid,
        frame,
        frame_id: int,
        session_id: str,
        timestamp_ms: Optional[int] = None,
    ) -> dict:
        session = self._get_session_or_none(uid, session_id)
        if session is None:
            return self._missing_session_response(uid, session_id)
        try:
            return session.process_frame(
                frame=frame,
                frame_id=int(frame_id),
                timestamp_ms=int(timestamp_ms if timestamp_ms is not None else time.time() * 1000),
            )
        except Exception as exc:
            return session.fail(f"处理视频帧失败: {exc}", frame_id=frame_id)

    def jump_status(self, uid, session_id: str) -> dict:
        session = self._get_session_or_none(uid, session_id)
        if session is None:
            return self._missing_session_response(uid, session_id)
        return session.status_response()

    def jump_stop(self, uid, session_id: str) -> dict:
        session = self._get_session_or_none(uid, session_id)
        if session is None:
            return self._missing_session_response(uid, session_id)
        return session.stop()

    def _get_session_or_none(self, uid, session_id: Optional[str]) -> Optional[FastReadyWebSession]:
        uid = str(uid)
        with self._store_lock:
            session = self._sessions.get(uid)
        if session is None:
            return None
        if not session_id or session.session_id != session_id:
            return None
        return session

    @staticmethod
    def _missing_session_response(uid, session_id) -> dict:
        return {
            "ok": False,
            "uid": str(uid) if uid is not None else None,
            "session_id": session_id,
            "frame_id": None,
            "state": "FAILED",
            "message": "测试会话不存在或已过期",
            "control": {"pause": True, "pause_frame_id": None},
            "painting": [],
            "score_cm": None,
            "error": "测试会话不存在或已过期",
        }


__all__ = ["JUMP"]
