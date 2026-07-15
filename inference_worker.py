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

    try:
        import tensorflow as tf
        model_path = PROJECTS[project_type]['model_path']
        log.info(f"加载模型: {model_path}")
        model = tf.keras.models.load_model(model_path)
        log.info("模型加载完成")
    except Exception as e:
        log.error(f"模型加载失败: {e}")
        sys.exit(1)

    idle_start = time.time()

    while True:
        try:
            task = task_queue.get(timeout=IDLE_TIMEOUT)
            idle_start = time.time()

            shm_name, user_id, frame_id = task

            shm, frame = SharedMemoryPool.attach(shm_name, SHAPE)
            if frame is None:
                log.error(f"无法读取共享内存 {shm_name}")
                shm_pool.release_block(shm_name)
                continue

            input_data = np.expand_dims(frame.astype(np.float32) / 255.0, axis=0)
            keypoints = model.predict(input_data)
            shm.close()

            eval_queue.put((shm_name, user_id, keypoints.tolist()))

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
