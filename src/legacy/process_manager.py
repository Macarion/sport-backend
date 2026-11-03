# process_manager.py
import subprocess
import threading
import signal
import sys
from pathlib import Path

_lock = threading.Lock()
_proc = None                 # 保存单例进程句柄
_SCRIPT = Path(__file__).resolve().parent / "new_situp.py"

def start() -> None:
    """在后台启动脚本；若已启动则忽略。"""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:       # 仍在运行
            return
        # 建议用绝对路径，避免工作目录问题
        _proc = subprocess.Popen(
            [sys.executable, str(_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )

def stop() -> None:
    """终止脚本；若未运行则忽略。"""
    global _proc
    with _lock:
        if _proc is None or _proc.poll() is not None:        # 已退出
            _proc = None
            return
        if sys.platform == "win32":
            _proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            _proc.send_signal(signal.SIGTERM)
        _proc.wait(timeout=10)                               # 最多等待 10 s
        _proc = None