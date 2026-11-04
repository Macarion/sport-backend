# worker/run_user_worker.py
"""用户进程入口（spawn target）：一个体测项目的推理 + 评测全链路，支持驻留复用（ADR-0003）。

消息协议：
  out_q（单向出口）：
    {'type': 'phase', 'phase': 'ready'}                       会话就绪（load/reset 完成后）
    {'type': 'result', 'frame_id', 'slot', 'data': dict|None}  每帧必回，slot 回执供 web 释放
    {'type': 'finished'}                                      响应停止信号
    {'type': 'error', 'message'}                              启动/处理失败，随后退出
  in_q（全部为 dict 信封，mp.Queue 底层 pickle）：
    {'type': 'start'}            开始新会话（冷启动与复用同路径，worker reset 后回 ready）
    {'type': 'stop'}             停止：会话中 → 回 finished 并转入驻留；驻留中 → 退出进程
    {'type': 'task', 'task': {…}} 帧任务体（SportTask.to_dict() 白名单字段；帧本体在
                                  shm_name 指向的共享内存中，绝不经管道传输）

生命周期：load（每进程一次）→ [等 start → reset → ready → 帧循环 → finished] 循环复用
→ 驻留窗口（IDLE_TIMEOUT）内等下一个 start，超时退出。
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
        sport.load()
        log.info('模型加载完成 耗时%.1fs', time.time() - t0)

        running = True
        while running:
            # ---- 等待会话开始（驻留窗口也在此计时）----
            try:
                raw = in_q.get(timeout=IDLE_TIMEOUT)
            except queue.Empty:
                log.info('驻留超过 %d 秒无新会话，退出', IDLE_TIMEOUT)
                break
            mtype = raw.get('type')
            if mtype == 'stop':
                log.info('驻留期间收到停止信号，退出')
                break
            if mtype != 'start':
                log.warning('会话未开始时收到 %s，忽略', mtype)
                continue

            sport.reset()
            out_q.put({'type': 'phase', 'phase': 'ready'})
            log.info('会话就绪')

            # ---- 会话帧循环 ----
            while True:
                # 会话空闲比驻留窗口多留 10 秒：让 web 的停止信号先到，收口走 finished 而非超时
                try:
                    raw = in_q.get(timeout=IDLE_TIMEOUT + 10)
                except queue.Empty:
                    log.info('会话空闲超过 %d 秒，退出', IDLE_TIMEOUT + 10)
                    running = False
                    break
                mtype = raw.get('type')
                if mtype == 'task':
                    task = SportTask.from_dict(raw['task'])
                    frame_id = task.get_frame_id()
                    with task.link_frame():
                        data = sport.process(task.get_frame(), task.get_config(),
                                             frame_id, task.get_uid())
                    out_q.put({'type': 'result', 'frame_id': frame_id,
                               'slot': task.get_shm_name(), 'data': data})
                elif mtype == 'stop':
                    out_q.put({'type': 'finished'})
                    break               # 回到外层：驻留等待下一个 start
                elif mtype == 'start':
                    log.warning('会话进行中收到重复 start，忽略')
                else:
                    log.warning('未知消息类型: %s', mtype)
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
                sport.close()
            except Exception as e:
                log.error('close 失败: %s', e)
        log.info('用户进程退出')
