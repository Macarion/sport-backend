# inference_worker.py
import os
import time
import queue
import multiprocessing as mp
from .config import IDLE_TIMEOUT, SPORTS, get_logger
from .shared_memory_pool import SharedMemoryPool
from .task import SportTask


def run_inference_worker(sport_type, task_queue, shm_pool, eval_queue):
    mp.current_process().name = f"Inference-{sport_type}"
    log = get_logger(f'Inference.{sport_type}')
    log.info("进程启动，PID=%d", os.getpid())

    inferencer = None

    match sport_type:
        case 'situp':
            from .sport_situp import Inferencer
            inferencer = Inferencer()
        case 'pullup':
            from .sport_pullup import Inferencer
            inferencer = Inferencer()
        case _:
            raise ValueError(f"未知的运动类型: {sport_type}")

    log.info("模型加载完成")

    idle_start = time.time()

    while True:
        try:
            task = SportTask.from_bytes(task_queue.get(timeout=IDLE_TIMEOUT + 0.1))
            idle_start = time.time()

            # 检查用户是否主动结束
            if task.is_finished():
                eval_queue.put(task)
                continue

            with task.link_frame():
                data = inferencer.process(task)

            task.set_data(data)
            eval_queue.put(task)

        except queue.Empty:
            elapsed = time.time() - idle_start
            if elapsed >= IDLE_TIMEOUT:
                log.info("空闲超过 %d 秒，进程自毁", IDLE_TIMEOUT)
                break
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            log.error("推理异常: %s", e)

    log.info("进程退出")
