# evaluation_worker.py
import os
import queue
from config import FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, get_logger
from shared_memory_pool import SharedMemoryPool

log = get_logger('Evaluation')

SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


def run_evaluation_worker(eval_queue, result_queue, shm_pool):
    log.info("进程启动，PID=%d", os.getpid())

    from sport_situp import Evaluator as SitupEvaluator

    user_sessions = {}

    while True:
        try:
            shm_name, user_id, data, project_type, config = eval_queue.get(timeout=1)

            # 检查用户是否主动结束
            if shm_name is None and data is None:
                if user_id in user_sessions:
                    del user_sessions[user_id]
                    result_queue.put((None, user_id, None))
                    log.debug("用户 %s 主动结束", user_id)
                continue

            if shm_name is None:
                log.error("无效的共享内存名称: %s, 用户ID: %s, 数据: %s, 配置: %s", shm_name, user_id, data, config)
            shm, frame = SharedMemoryPool.attach(shm_name, SHAPE)
            if frame is None:
                shm_pool.release_block(shm_name)
                continue

            if user_id not in user_sessions:
                match project_type:
                    case "situp":
                        evaluator = SitupEvaluator()
                        user_sessions[user_id] = evaluator
                    case _:
                        raise ValueError(f"未知的运动类型: {project_type}")
            else:
                evaluator = user_sessions[user_id]
            result = evaluator.process(frame, data, config)

            shm.close()

            result_queue.put((shm_name, user_id, result))

        except queue.Empty:
            continue
        except (KeyboardInterrupt, EOFError):
            break

    log.info("进程退出")

