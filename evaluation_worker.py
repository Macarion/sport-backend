# evaluation_worker.py
import os
import queue
from config import FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, get_logger
from shared_memory_pool import SharedMemoryPool

log = get_logger('Evaluation')

SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


def run_evaluation_worker(eval_queue, result_queue, shm_pool):
    log.info("进程启动，PID=%d", os.getpid())

    from sport_situp import Evaluator

    user_sessions = {}

    while True:
        try:
            shm_name, user_id, data, config = eval_queue.get(timeout=1)

            # 检查用户是否主动结束
            if shm_name is None and data is None:
                if user_id in user_sessions:
                    del user_sessions[user_id]
                    result_queue.put((None, user_id, None))
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
            result = evaluator.process(frame, data, config)

            shm.close()

            result_queue.put((shm_name, user_id, result))

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break

    log.info("进程退出")

