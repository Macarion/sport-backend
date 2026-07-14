# inference_worker.py
import os
import sys
import time
import queue
import numpy as np
import multiprocessing as mp
from config import IDLE_TIMEOUT, PROJECTS

# 注意：tensorflow/onnx 等重型库在子进程内部导入，主进程不加载
def run_inference_worker(project_type, manager_client, shm_pool_client, eval_queue):
    """
    推理子进程的主函数。
    project_type: 项目名称（如 'situp'）
    manager_client: 连接 Launcher 管理器，用于获取任务队列
    shm_pool_client: 共享内存池代理
    eval_queue: 连接评测进程的队列
    """
    # 1. 设置进程名称（便于调试）
    mp.current_process().name = f"Inference-{project_type}"
    
    # 2. 延迟导入重型库（只在子进程加载）
    print(f"[{project_type}] 进程启动，PID={os.getpid()}")
    try:
        import tensorflow as tf
        from config import PROJECTS
        model_path = PROJECTS[project_type]['model_path']
        print(f"[{project_type}] 加载模型: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"[{project_type}] 模型加载完成！")
    except Exception as e:
        print(f"[{project_type}] 模型加载失败: {e}")
        sys.exit(1)

    # 3. 连接 Launcher 获取该项目的专属任务队列
    #    由于 Manager 提供了跨进程队列，直接在子进程中连接
    task_queue = manager_client.get_task_queue(project_type)
    
    # 4. 初始化空闲计时器
    idle_start = time.time()
    
    # 5. 主循环
    while True:
        try:
            # 阻塞超时获取任务（超时时间 = 空闲超时阈值）
            task = task_queue.get(timeout=IDLE_TIMEOUT)
            # 收到任务，重置空闲计时器
            idle_start = time.time()
            
            # 解析任务： (shm_name, user_id, frame_id)
            shm_name, user_id, frame_id = task
            
            # 从共享内存读取帧
            frame = shm_pool_client.get_array(shm_name)
            if frame is None:
                print(f"[{project_type}] 错误：无法读取共享内存 {shm_name}")
                continue
            
            # ---------- 执行推理 ----------
            # 假设模型接收 (1, H, W, 3) 的归一化输入
            input_data = np.expand_dims(frame.astype(np.float32) / 255.0, axis=0)
            keypoints = model.predict(input_data)  # 假设返回关键点坐标
            
            # 注意：关键点坐标可以在 frame 上绘制，也可以作为 JSON 传递
            # 这里我们直接在共享内存帧上绘制骨骼点（用于后续评测进程叠加报表）
            # draw_keypoints(frame, keypoints)  # 假设有此函数
            
            # ---------- 将结果推给评测进程 ----------
            # 传递：共享内存名称、用户ID、关键点数据（转为list便于传输）
            eval_queue.put((shm_name, user_id, keypoints.tolist()))
            
        except queue.Empty:
            # 超时未收到任务，检查空闲时长
            elapsed = time.time() - idle_start
            if elapsed >= IDLE_TIMEOUT:
                print(f"[{project_type}] 空闲超过 {IDLE_TIMEOUT} 秒，进程自毁。")
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{project_type}] 推理异常: {e}")
            time.sleep(1)
    
    print(f"[{project_type}] 进程退出")
    sys.exit(0)