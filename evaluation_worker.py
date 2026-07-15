# evaluation_worker.py
import os
import sys
import time
import queue
from collections import defaultdict
from config import FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, get_logger
from shared_memory_pool import SharedMemoryPool

log = get_logger('Evaluation')

SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


def run_evaluation_worker(eval_queue, result_queue, shm_pool):
    log.info(f"进程启动，PID={os.getpid()}")

    from sport_situp import Evaluator

    user_sessions = {}

    while True:
        try:
            shm_name, user_id, data = eval_queue.get(timeout=1)
            if shm_name is None and data is None:
                if user_id in user_sessions:
                    del user_sessions[user_id]
                    continue

            shm, frame = SharedMemoryPool.attach(shm_name, SHAPE)
            if frame is None:
                shm_pool.release_block(shm_name)
                continue

            if user_id not in user_sessions:
                evaluator = Evaluator()
                user_sessions[user_id] = evaluator
            else:
                evaluator = user_sessions[user_id]
            result = evaluator.process(frame, data)

            shm.close()

            result_queue.put((shm_name, user_id, result))

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"错误: {e}")
            time.sleep(1)

    log.info("进程退出")

