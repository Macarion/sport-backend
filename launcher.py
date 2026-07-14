# launcher.py
import os
import sys
import time
import subprocess
import signal
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import atexit

from config import (
    PROJECTS, USERS_PER_PROCESS, MAX_PROCESSES_PER_PROJECT,
    IDLE_TIMEOUT, SCAN_INTERVAL, MANAGER_HOST, MANAGER_PORT, AUTH_KEY
)
from shared_memory_pool import SharedMemoryPool

# 强制使用 spawn（Windows/Linux 统一）
mp.set_start_method('spawn', force=True)

class QueueManager(BaseManager):
    pass

class Launcher:
    def __init__(self):
        self.running = True
        self.processes = {}  # {project_type: [pid1, pid2, ...]}
        self.user_counts = mp.Manager().dict()  # {project_type: count}
        self.project_queues = {}  # {project_type: Queue}
        self.command_queue = None  # 接收 django 命令的队列
        
        # 1. 初始化共享内存池（主进程持有，子进程通过代理访问）
        self.shm_pool = SharedMemoryPool()
        
        # 2. 启动 Manager 服务器（提供跨进程队列）
        self.start_manager_server()
        
        # 3. 注册信号处理
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        atexit.register(self.cleanup)
        
        # 4. 初始化各项目的用户计数和队列
        for proj in PROJECTS.keys():
            self.user_counts[proj] = 0
            # 创建该项目的专属任务队列（通过 manager 提供）
            self.project_queues[proj] = self.manager.get_task_queue(proj)
        
        # 5. 启动评测进程（单例）
        self.start_evaluation_process()
        
        # 6. 启动 Django（Daphne）进程
        # self.start_django()
    
    def start_manager_server(self):
        """启动跨进程队列管理器"""
        
        # 注册队列创建方法
        self.manager = QueueManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=AUTH_KEY)
        self.manager.start()
        
        # 为每个项目注册独立的任务队列
        for proj in PROJECTS.keys():
            q = mp.Queue()
            QueueManager.register(f'get_task_queue_{proj}', callable=lambda q=q: q)
        
        # 注册评测结果队列和命令队列
        self.result_queue = mp.Queue()
        self.send_to_ws_queue = mp.Queue()
        self.cmd_queue = mp.Queue()
        QueueManager.register('get_result_queue', callable=lambda: self.result_queue)
        QueueManager.register('get_send_queue', callable=lambda: self.send_to_ws_queue)
        QueueManager.register('get_command_queue', callable=lambda: self.cmd_queue)
        
        print(f"[Launcher] Manager 服务已启动: {MANAGER_HOST}:{MANAGER_PORT}")
    
    def start_django(self):
        """启动 Django + Daphne 子进程（subprocess）"""
        print("[Launcher] 启动 Django 服务...")
        cmd = [
            sys.executable, "-m", "daphne",
            "-p", "8090",
            "-b", "0.0.0.0",
            "backend.asgi:application"
        ]
        # 将 Manager 地址写入环境变量，供 Django 进程连接
        env = os.environ.copy()
        env['MANAGER_HOST'] = MANAGER_HOST
        env['MANAGER_PORT'] = str(MANAGER_PORT)
        env['AUTH_KEY'] = AUTH_KEY.hex()
        
        self.django_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        print(f"[Launcher] Django PID: {self.django_proc.pid}")
    
    def start_evaluation_process(self):
        """启动评测计数进程"""
        print("[Launcher] 启动评测进程...")
        from evaluation_worker import run_evaluation_worker
        # 评测进程需要通过代理访问共享内存池和队列
        # 注意：由于 spawn 模式，需将 callable 和参数 pickle，这里将对象转为代理或使用 Manager
        # 简化：传递共享内存池的名称列表
        shm_names = [b['name'] for b in self.shm_pool.blocks]
        proc = mp.Process(
            target=run_evaluation_worker,
            args=(self.result_queue, self.send_to_ws_queue, shm_names)
        )
        proc.start()
        self.eval_process = proc
        print(f"[Launcher] 评测进程 PID: {proc.pid}")
    
    def spawn_inference_process(self, project_type):
        """创建一个新的推理子进程"""
        from inference_worker import run_inference_worker
        
        # 传递队列代理和共享内存名称列表给子进程
        task_queue = self.project_queues[project_type]
        shm_names = [b['name'] for b in self.shm_pool.blocks]
        
        proc = mp.Process(
            target=run_inference_worker,
            args=(project_type, task_queue, shm_names, self.result_queue)
        )
        proc.start()
        
        if project_type not in self.processes:
            self.processes[project_type] = []
        self.processes[project_type].append(proc)
        print(f"[Launcher] 创建推理进程: {project_type}, PID={proc.pid}")
        return proc
    
    def scale_loop(self):
        """定时检查并执行扩缩容"""
        while self.running:
            try:
                # 处理来自 Django 的命令（用户注册/注销）
                while not self.cmd_queue.empty():
                    cmd, proj, delta = self.cmd_queue.get_nowait()
                    if proj in self.user_counts:
                        self.user_counts[proj] = max(0, self.user_counts[proj] + delta)
                        print(f"[Launcher] 用户计数更新: {proj} = {self.user_counts[proj]}")
                
                # 遍历所有项目
                for proj in PROJECTS.keys():
                    current_users = self.user_counts.get(proj, 0)
                    current_processes = len(self.processes.get(proj, []))
                    
                    # 计算期望进程数
                    target = (current_users + USERS_PER_PROCESS - 1) // USERS_PER_PROCESS
                    target = min(target, MAX_PROCESSES_PER_PROJECT)
                    target = max(1, target) if current_users > 0 else 0  # 无用户则期望为0
                    
                    # 扩容：当前 < 期望
                    while current_processes < target:
                        self.spawn_inference_process(proj)
                        current_processes += 1
                    
                    # 缩容：当前 > 期望（不主动杀，交给子进程空闲超时自毁）
                    # 但我们需清理已死亡的进程
                    alive = []
                    for p in self.processes.get(proj, []):
                        if p.is_alive():
                            alive.append(p)
                        else:
                            print(f"[Launcher] 检测到进程死亡: {proj} PID={p.pid}")
                    self.processes[proj] = alive
                    
            except Exception as e:
                print(f"[Launcher] 调度循环错误: {e}")
            
            time.sleep(SCAN_INTERVAL)
    
    def cleanup(self):
        """清理所有资源"""
        print("\n[Launcher] 正在清理...")
        self.running = False
        
        # 终止所有子进程
        for proj, procs in self.processes.items():
            for p in procs:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
                    if p.is_alive():
                        p.kill()
        if hasattr(self, 'django_proc') and self.django_proc.poll() is None:
            self.django_proc.terminate()
            self.django_proc.wait(timeout=5)
        if hasattr(self, 'eval_process') and self.eval_process.is_alive():
            self.eval_process.terminate()
            self.eval_process.join(timeout=2)
        
        # 清理共享内存
        self.shm_pool.cleanup()
        
        # 关闭 Manager
        if hasattr(self, 'manager'):
            self.manager.shutdown()
        
        print("[Launcher] 清理完成")
    
    def shutdown(self, signum=None, frame=None):
        self.cleanup()
        sys.exit(0)
    
    def run(self):
        """主循环"""
        print("[Launcher] 系统启动中...")
        try:
            self.scale_loop()
        except KeyboardInterrupt:
            self.shutdown()


if __name__ == "__main__":
    launcher = Launcher()
    launcher.run()