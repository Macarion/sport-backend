# worker/run_user_worker.py
"""用户进程入口（spawn target）：一个体测会话的推理 + 评测全链路。

消息协议（out_q 单一出口）：
  {'type': 'phase', 'phase': 'ready'}                       模型加载完成
  {'type': 'result', 'frame_id', 'slot', 'data': dict|None} 每帧必回，slot 回执供 web 释放
  {'type': 'finished'}                                      响应停止信号，随后退出
  {'type': 'error', 'message'}                              启动/处理失败，随后退出
web -> in_q：SportTask 字节（正常帧带 shm_name/frame_shape/config；停止为 finished=True）。
"""
import os
import queue
import time
import multiprocessing as mp

from core.config import IDLE_TIMEOUT
from core.log import get_logger
from core.registry import load_sport
from core.task import SportTask


def run_user_worker(sport_type, in_q, out_q):
    mp.current_process().name = f'User-{sport_type}'
    log = get_logger(f'Worker.{sport_type}')
    log.info('用户进程启动 PID=%d', os.getpid())

    sport = None
    try:
        sport_cls = load_sport(sport_type)
        sport = sport_cls()
        t0 = time.time()
        sport.start()
        log.info('模型与状态机加载完成 耗时%.1fs', time.time() - t0)
        out_q.put({'type': 'phase', 'phase': 'ready'})

        while True:
            try:
                raw = in_q.get(timeout=IDLE_TIMEOUT)
            except queue.Empty:
                log.info('空闲超过 %d 秒，退出', IDLE_TIMEOUT)
                break
            task = SportTask.from_bytes(raw)
            if task.is_finished():
                out_q.put({'type': 'finished'})
                break
            frame_id = task.get_frame_id()
            with task.link_frame():
                data = sport.process(task.get_frame(), task.get_config(),
                                     frame_id, task.get_uid())
            out_q.put({'type': 'result', 'frame_id': frame_id,
                       'slot': task.get_shm_name(), 'data': data})
    except (KeyboardInterrupt, EOFError):
        log.info('收到中断或管道关闭，退出')
    except Exception as e:
        log.error('用户进程失败: %s', e, exc_info=True)
        try:
            out_q.put({'type': 'error', 'message': str(e)})
        except Exception:
            pass
    finally:
        if sport is not None:
            try:
                sport.stop()
            except Exception as e:
                log.error('stop 失败: %s', e)
        log.info('用户进程退出')
