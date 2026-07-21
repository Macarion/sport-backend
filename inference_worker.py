# inference_worker.py
import os
import sys
import time
import queue
import multiprocessing as mp
from config import IDLE_TIMEOUT, PROJECTS, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, get_logger
from shared_memory_pool import SharedMemoryPool

SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


def run_inference_worker(project_type, task_queue, shm_pool, eval_queue):
    mp.current_process().name = f"Inference-{project_type}"
    log = get_logger(f'Inference.{project_type}')
    log.info("进程启动，PID=%d", os.getpid())

    inferencer = None

    match project_type:
        case 'situp':
            from sport_situp import Inferencer
            inferencer = Inferencer()
        case _:
            raise ValueError(f"未知的运动类型: {project_type}")

    log.info("模型加载完成")

    idle_start = time.time()

    while True:
        try:
            shm_name, user_id, frame_id, config = task_queue.get(timeout=IDLE_TIMEOUT)
            idle_start = time.time()

            # 检查用户是否主动结束
            if frame_id < 0:
                eval_queue.put((None, user_id, None, project_type, config))
                continue

            shm, frame = SharedMemoryPool.attach(shm_name, SHAPE)
            if frame is None:
                log.error("无法读取共享内存 %s", shm_name)
                shm_pool.release_block(shm_name)
                continue

            result = inferencer.process(frame_id, frame, config)
            shm.close()

            eval_queue.put((shm_name, user_id, result, project_type, config))

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
