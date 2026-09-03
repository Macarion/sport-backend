# evaluation_worker.py
import os
import queue
from .config import SPORTS, get_logger
from .shared_memory_pool import SharedMemoryPool
from .task import SportTask

log = get_logger('Evaluation')

def run_evaluation_worker(eval_queue, result_queue, shm_pool):
    log.info("进程启动，PID=%d", os.getpid())

    from .sport_situp import Evaluator as SitupEvaluator

    user_sessions = {}

    while True:
        try:
            task: SportTask = eval_queue.get(timeout=1)

            # 检查用户是否主动结束
            uid = task.get_uid()
            frame_id = task.get_frame_id()
            sport_type = task.get_sport_type()

            # 检查用户是否主动结束
            if task.is_finished():
                if uid in user_sessions:
                    del user_sessions[uid]
                    result_queue.put(task.to_bytes())

            evaluator = None
            if frame_id == 0 and uid not in user_sessions:
                match sport_type:
                    case "situp":
                        evaluator = SitupEvaluator()
                        log.info("创建新会话，用户 %s", uid)
                        user_sessions[uid] = evaluator
                    case _:
                        raise ValueError(f"未知的运动类型: {sport_type}")
            elif uid in user_sessions:
                evaluator = user_sessions[uid]

            if evaluator is not None:
                with task.link_frame():
                    data = evaluator.process(task)
            else:
                # 如果用户不在会话中，说明该帧是由多进程推理的乱序执行导致用户结束测试后的残留帧，忽略该帧
                data = None

            task.set_data(data)
            result_queue.put(task.to_bytes())

        except queue.Empty:
            continue
        except (KeyboardInterrupt, EOFError):
            break

    log.info("进程退出")

