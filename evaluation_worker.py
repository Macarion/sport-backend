# evaluation_worker.py
import os
import sys
import time
from collections import defaultdict

def run_evaluation_worker(result_queue, send_to_ws_queue, shm_pool_client):
    """
    评测计数进程（单例）。
    接收来自所有推理进程的关键点数据，进行动作计数，并将结果帧发回 WebSocket。
    """
    print(f"[Evaluation] 进程启动，PID={os.getpid()}")
    
    # 维护每个用户的状态（用于计数）
    user_sessions = defaultdict(lambda: {'count': 0, 'last_pose': None})
    
    while True:
        try:
            shm_name, user_id, keypoints_list = result_queue.get(timeout=1)
            
            # 从共享内存读取帧（此帧已被推理进程绘制了骨骼点）
            frame = shm_pool_client.get_array(shm_name)
            if frame is None:
                continue
            
            # ---------- 执行计数逻辑 ----------
            # 这里简化：调用你的计数函数
            # new_count = count_action(user_id, keypoints_list, user_sessions)
            new_count = 10  # 模拟计数
            
            # 在共享内存帧上叠加计数文本（绘制报表）
            # cv2.putText(frame, f"Count: {new_count}", (10,30), ...)
            
            # ---------- 将结果帧和计数数据回传给 WebSocket ----------
            # 注意：此时 frame 仍然在共享内存中，只需将 shm_name 和计数数据发回
            send_to_ws_queue.put((shm_name, user_id, {'count': new_count}))
            
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Evaluation] 错误: {e}")
            time.sleep(1)
    
    print("[Evaluation] 进程退出")