# inference_worker.py
import os
import sys
import time
import queue
import numpy as np
import multiprocessing as mp
from config import IDLE_TIMEOUT, PROJECTS, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, get_logger
from shared_memory_pool import SharedMemoryPool

SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


def run_inference_worker(project_type, task_queue, shm_pool, eval_queue):
    mp.current_process().name = f"Inference-{project_type}"
    log = get_logger(f'Inference.{project_type}')
    log.info(f"进程启动，PID={os.getpid()}")

    worker = None

    if project_type == 'situp':
        from sport_situp import PointDetectionWorker
        worker = PointDetectionWorker()
        log.info("模型加载完成")

    if worker is None:
        log.error("未找到对应的项目")
        return

    idle_start = time.time()

    while True:
        try:
            task = task_queue.get(timeout=IDLE_TIMEOUT)
            idle_start = time.time()

            shm_name, user_id, frame_id = task
            if frame_id < 0:
                eval_queue.put((None, user_id, None))
                continue

            shm, frame = SharedMemoryPool.attach(shm_name, SHAPE)
            if frame is None:
                log.error(f"无法读取共享内存 {shm_name}")
                shm_pool.release_block(shm_name)
                continue

            # input_data = np.expand_dims(frame.astype(np.float32) / 255.0, axis=0)
            result = worker.process(frame_id, frame)
            shm.close()

            eval_queue.put((shm_name, user_id, result))

        except queue.Empty:
            elapsed = time.time() - idle_start
            if elapsed >= IDLE_TIMEOUT:
                log.info(f"空闲超过 {IDLE_TIMEOUT} 秒，进程自毁")
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"推理异常: {e}")
            time.sleep(1)

    log.info("进程退出")
    sys.exit(0)
