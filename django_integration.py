# django_integration.py
# 将此代码放入 Django 的 views.py 或 consumers.py 中

import os
from multiprocessing.managers import BaseManager
import time

# 从环境变量获取 Manager 连接信息
MANAGER_HOST = os.environ.get('MANAGER_HOST', '127.0.0.1')
MANAGER_PORT = int(os.environ.get('MANAGER_PORT', 50000))
AUTH_KEY = bytes.fromhex(os.environ.get('AUTH_KEY', '626f6479746573745f7365637265745f6b6579'))

class QueueManager(BaseManager):
    pass

# 注册所有需要的队列（与 Launcher 保持一致）
QueueManager.register('get_task_queue')
QueueManager.register('get_command_queue')
QueueManager.register('get_send_queue')

# 连接管理器（全局单例，在 Django 启动时初始化一次）
def get_manager():
    manager = QueueManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=AUTH_KEY)
    manager.connect()
    return manager

# ---------- 在 WebSocket Consumer 中使用 ----------
# 假设你在 Django Channels 的 AsyncWebsocketConsumer 中
class VideoConsumer:
    async def connect(self):
        self.user_id = self.scope['url_route']['kwargs']['user_id']
        self.project = self.scope['url_route']['kwargs']['project']  # 如 'situp'
        
        # 1. 通知 Launcher 增加该项目的用户计数
        manager = get_manager()
        cmd_queue = manager.get_command_queue()
        cmd_queue.put(('register', self.project, 1))
        
        # 2. 获取该项目的任务队列（用于发送视频帧）
        self.task_queue = manager.get_task_queue(self.project)
        # 3. 获取结果队列（用于接收评测结果）
        self.send_queue = manager.get_send_queue()
        
        await self.accept()
    
    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # 假设收到的是原始视频帧 bytes (需编码为 jpeg 或原始 RGB)
            # 实际项目中需要从前端传 SharedMemory 名称，这里简化演示
            # 正确流程：前端传帧 -> Django 解码 -> 写入共享内存
            # 这里调用 shared_memory_pool 的 acquire_block 申请内存
            # 写入后将 (shm_name, user_id) 放入 task_queue
            pass

    async def disconnect(self, close_code):
        # 通知 Launcher 减少用户计数
        manager = get_manager()
        cmd_queue = manager.get_command_queue()
        cmd_queue.put(('unregister', self.project, -1))