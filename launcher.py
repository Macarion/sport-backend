# launcher.py
import os
import sys
import subprocess
import signal
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import atexit
import queue

from config import (
    PROJECTS, USERS_PER_PROCESS, MAX_PROCESSES_PER_PROJECT,
    IDLE_TIMEOUT, MANAGER_HOST, MANAGER_PORT, AUTH_KEY, get_logger
)
from shared_memory_pool import SharedMemoryPool, SHM_NAMESPACE

log = get_logger('Launcher')

mp.set_start_method('spawn', force=True)

# 模块级状态（BaseManager callable 必须引用模块级对象）
_result_queue = None
_send_to_ws_queue = None
_cmd_queue = None
_project_queues = {}


def _get_result_queue():
    return _result_queue


def _get_send_queue():
    return _send_to_ws_queue


def _get_command_queue():
    return _cmd_queue


def _get_task_queue(proj):
    return _project_queues[proj]


class QueueManager(BaseManager):
    pass


_shm_pool_ref = None


def _get_shm_pool():
    return _shm_pool_ref


QueueManager.register('get_result_queue', callable=_get_result_queue)
QueueManager.register('get_send_queue', callable=_get_send_queue)
QueueManager.register('get_command_queue', callable=_get_command_queue)
QueueManager.register('get_task_queue', callable=_get_task_queue)
QueueManager.register('get_shm_pool', callable=_get_shm_pool)


class Launcher:
    def __init__(self):
        global _result_queue, _send_to_ws_queue, _cmd_queue, _project_queues, _shm_pool_ref

        self.running = True
        self.processes = {}
        self.user_counts = {}

        _result_queue = mp.Queue()
        _send_to_ws_queue = mp.Queue()
        _cmd_queue = mp.Queue()

        self.result_queue = _result_queue
        self.send_to_ws_queue = _send_to_ws_queue
        self.cmd_queue = _cmd_queue

        _project_queues = {}
        for proj in PROJECTS.keys():
            _project_queues[proj] = mp.Queue()
        self.project_queues = _project_queues

        self.shm_pool = SharedMemoryPool(namespace=SHM_NAMESPACE)
        _shm_pool_ref = self.shm_pool

        self.start_manager_server()

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        atexit.register(self.cleanup)

        for proj in PROJECTS.keys():
            self.user_counts[proj] = 0

        self.start_evaluation_process()

    def start_manager_server(self):
        self.manager = QueueManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=AUTH_KEY)
        self.manager.start()
        log.info(f"Manager 服务已启动: {MANAGER_HOST}:{MANAGER_PORT}")

    def start_django(self):
        log.info("启动 Django 服务...")
        cmd = [
            sys.executable, "-m", "daphne",
            "-p", "8090",
            "-b", "0.0.0.0",
            "backend.asgi:application"
        ]
        env = os.environ.copy()
        env['MANAGER_HOST'] = MANAGER_HOST
        env['MANAGER_PORT'] = str(MANAGER_PORT)
        env['AUTH_KEY'] = AUTH_KEY.hex()

        self.django_proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        log.info(f"Django PID: {self.django_proc.pid}")

    def start_evaluation_process(self):
        log.info("启动评测进程...")
        from evaluation_worker import run_evaluation_worker
        proc = mp.Process(
            target=run_evaluation_worker,
            args=(self.result_queue, self.send_to_ws_queue, self.shm_pool)
        )
        proc.start()
        self.eval_process = proc
        log.info(f"评测进程 PID: {proc.pid}")

    def spawn_inference_process(self, project_type):
        from inference_worker import run_inference_worker
        task_queue = self.project_queues[project_type]

        proc = mp.Process(
            target=run_inference_worker,
            args=(project_type, task_queue, self.shm_pool, self.result_queue)
        )
        proc.start()

        if project_type not in self.processes:
            self.processes[project_type] = []
        self.processes[project_type].append(proc)
        log.info(f"创建推理进程: {project_type}, PID={proc.pid}")
        return proc

    def handle_command(self, cmd, proj, delta):
        if proj not in self.user_counts:
            return
        self.user_counts[proj] = max(0, self.user_counts[proj] + delta)
        log.info(f"用户计数更新: {proj} = {self.user_counts[proj]}")
        self._adjust_processes(proj)

    def _adjust_processes(self, proj):
        current_users = self.user_counts.get(proj, 0)
        current_processes = len(self.processes.get(proj, []))

        target = (current_users + USERS_PER_PROCESS - 1) // USERS_PER_PROCESS
        target = min(target, MAX_PROCESSES_PER_PROJECT)
        target = max(1, target) if current_users > 0 else 0

        while current_processes < target:
            self.spawn_inference_process(proj)
            current_processes += 1

        alive = []
        for p in self.processes.get(proj, []):
            if p.is_alive():
                alive.append(p)
            else:
                log.info(f"检测到进程退出: {proj} PID={p.pid}")
        self.processes[proj] = alive

    def event_loop(self):
        log.info("进入事件驱动主循环...")
        while self.running:
            try:
                cmd, proj, delta = self.cmd_queue.get(timeout=1.0)
                self.handle_command(cmd, proj, delta)
            except queue.Empty:
                for proj in PROJECTS.keys():
                    self._adjust_processes(proj)
            except Exception as e:
                log.error(f"事件循环错误: {e}")

    def cleanup(self):
        if not self.running:
            return
        self.running = False
        log.info("正在清理...")

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

        self.shm_pool.cleanup()

        if hasattr(self, 'manager'):
            self.manager.shutdown()

        log.info("清理完成")

    def shutdown(self, signum=None, frame=None):
        self.cleanup()
        sys.exit(0)

    def run(self):
        log.info("系统启动中...")
        try:
            self.event_loop()
        except KeyboardInterrupt:
            self.shutdown()


if __name__ == "__main__":
    launcher = Launcher()
    launcher.run()
