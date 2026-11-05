# src/main.py
"""唯一入口（ADR-0002）：顶层只留轻量 import；guard 内程序化托管 daphne、清理收口、开机预热。

顶层必须保持轻——spawn 出的每个子进程都会重新执行本文件顶层（实测见 ADR-0002）。
"""
import atexit
import os
import signal
import sys
import time
from core.config import BACKEND_ROOT, DAPHNE_PORT, FACE_GATE_ENABLED, PREWARM_ENABLED
from core.log import get_logger

log = get_logger('main')
_cleaned = False


def _cleanup():
    """进程退出收口：销毁全部用户会话（等进程退出、回收共享内存槽）。"""
    global _cleaned
    if _cleaned:
        return
    _cleaned = True
    log.info('正在清理...')
    try:
        from api.user_manager import user_manager  # 重依赖仅在此刻加载
        user_manager.shutdown_all()
    except Exception as e:
        log.error('会话清理失败: %s', e)
    log.info('清理完成')


def _prewarm():
    """开机预热（一次性子进程内）：注册表全解析 + 逐项加载模型打热文件缓存。"""
    from core.registry import SPORTS, load_sport
    for key in sorted(SPORTS):
        t0 = time.time()
        try:
            sport = load_sport(key)()
            sport.load()
            sport.close()
            log.info('预热完成: %s 耗时%.1fs', key, time.time() - t0)
        except Exception as e:
            log.error('预热失败: %s -> %s', key, e)


def main():
    os.chdir(BACKEND_ROOT)  # 未锚定的相对路径（latest.jpg、auth.log 等）以 backend 根为基准；vendor 资产已走绝对路径

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    if PREWARM_ENABLED:
        prewarm_proc = mp.Process(target=_prewarm, daemon=True)
        prewarm_proc.start()
        log.info('预热进程已启动: PID=%d（日志带“预热”字样）', prewarm_proc.pid)
    else:
        log.info('预热已关闭（config.PREWARM_ENABLED=False），模型由首个会话懒加载')

    if FACE_GATE_ENABLED:
        from core.face_service import spawn_face_service
        spawn_face_service()   # 异步预载，不阻塞 daphne 启动

    from daphne.cli import CommandLineInterface
    cli = CommandLineInterface()
    try:
        cli.run(['-p', str(DAPHNE_PORT), 'backend.asgi:application'])
    finally:
        _cleanup()


if __name__ == '__main__':
    main()
