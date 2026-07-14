# shared_memory_pool.py
import sys
import time
from multiprocessing import shared_memory, Lock
import numpy as np
from config import SHM_POOL_SIZE, FRAME_BYTES, FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS

class SharedMemoryPool:
    """
    预分配固定数量的共享内存块，用于零拷贝传输视频帧。
    所有进程通过 attach 方式连接。
    """
    def __init__(self):
        self.lock = Lock()
        self.blocks = []
        self.in_use = [False] * SHM_POOL_SIZE
        self.shape = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)
        
        # 预创建共享内存块（Windows 下名称全局唯一，加 PID 防止冲突）
        pid = str(1234) # 实际应用中可用 os.getpid()，但这里由 launcher 传入
        print(f"[SHM] 正在初始化 {SHM_POOL_SIZE} 个共享内存块...")
        for i in range(SHM_POOL_SIZE):
            name = f"frame_{pid}_{i}"
            try:
                # 尝试创建（如果已存在则先清理，防止上次崩溃残留）
                shm = shared_memory.SharedMemory(name=name, create=True, size=FRAME_BYTES)
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
                shm = shared_memory.SharedMemory(name=name, create=True, size=FRAME_BYTES)
            
            # 初始化缓冲区为零
            shm.buf[:FRAME_BYTES] = b'\x00' * FRAME_BYTES
            self.blocks.append({
                'name': name,
                'shm': shm,
                'size': FRAME_BYTES
            })
        print("[SHM] 初始化完成。")

    def acquire_block(self, timeout=1.0):
        """申请一个空闲内存块，返回 (shm_name, numpy_array)"""
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                for i, used in enumerate(self.in_use):
                    if not used:
                        self.in_use[i] = True
                        block = self.blocks[i]
                        # 创建 numpy 视图（便于写入）
                        arr = np.ndarray(self.shape, dtype=np.uint8, buffer=block['shm'].buf)
                        return block['name'], arr
            time.sleep(0.001)
        raise TimeoutError("共享内存池已满，无可用块！")
    
    def release_block(self, shm_name):
        """归还内存块"""
        with self.lock:
            for i, block in enumerate(self.blocks):
                if block['name'] == shm_name:
                    self.in_use[i] = False
                    return True
        return False

    def get_array(self, shm_name):
        """根据名称获取 numpy 视图（供子进程读取）"""
        for block in self.blocks:
            if block['name'] == shm_name:
                return np.ndarray(self.shape, dtype=np.uint8, buffer=block['shm'].buf)
        raise ValueError(f"内存块 {shm_name} 不存在")

    def cleanup(self):
        """清理所有共享内存（Launcher 退出时调用）"""
        print("[SHM] 正在清理共享内存...")
        for block in self.blocks:
            try:
                block['shm'].close()
                block['shm'].unlink()
            except Exception as e:
                print(f"清理 {block['name']} 时出错: {e}")