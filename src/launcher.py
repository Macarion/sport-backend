# launcher.py
import os
import sys
import math
import queue
import signal
import atexit
import subprocess
import multiprocessing as mp
from .queue_manager import QueueManager

from .config import (
    SPORTS, USERS_PER_PROCESS, MAX_PROCESSES_PER_PROJECT,
    IDLE_TIMEOUT, MANAGER_HOST, MANAGER_PORT, AUTH_KEY, get_logger
)

log = get_logger('Launcher')

mp.set_start_method('spawn', force=True)

class Launcher:
    def __init__(self):
        self.running = True
        self.processes = {}
        self.user_counts = {}

        self.manager = QueueManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=AUTH_KEY)
        self.manager.start()

        self.eval_queue = self.manager.get_eval_queue()
        self.result_queue = self.manager.get_result_queue()
        self.cmd_queue = self.manager.get_command_queue()
        self.shm_pool = self.manager.get_shm_pool()

        self.task_queues = {}
        for proj in SPORTS.keys():
            self.task_queues[proj] = self.manager.get_task_queue(proj)

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        atexit.register(self.cleanup)

        for proj in SPORTS.keys():
            self.user_counts[proj] = 0

        self.start_django()

        self.start_evaluation_process()

    def start_django(self):
        from pathlib import Path

        log.info("启动 Django 服务...")
        cmd = [
            sys.executable, "-m", "daphne",
            "-p", "8090",
            "backend.asgi:application"
        ]
        env = os.environ.copy()
        env['MANAGER_HOST'] = MANAGER_HOST
        env['MANAGER_PORT'] = str(MANAGER_PORT)
        env['AUTH_KEY'] = AUTH_KEY.hex()

        self.django_proc = subprocess.Popen(
            cmd, env=env,
            cwd=Path(__file__).parent / "web_server",
            # stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        log.info("Django PID: %d", self.django_proc.pid)

    def start_evaluation_process(self):
        log.info("启动评测进程...")
        from .evaluation_worker import run_evaluation_worker
        proc = mp.Process(
            target=run_evaluation_worker,
            args=(self.eval_queue, self.result_queue, self.shm_pool),
            daemon=True
        )
        proc.start()
        self.eval_process = proc
        log.info("评测进程 PID: %d", proc.pid)

    def spawn_inference_process(self, sport_type):
        from .inference_worker import run_inference_worker
        task_queue = self.task_queues[sport_type]

        proc = mp.Process(
            target=run_inference_worker,
            args=(sport_type, task_queue, self.shm_pool, self.eval_queue),
            daemon=True
        )
        proc.start()

        if sport_type not in self.processes:
            self.processes[sport_type] = []
        self.processes[sport_type].append(proc)
        log.info("创建推理进程: %s, PID=%d", sport_type, proc.pid)
        return proc

    def handle_command(self, cmd, proj, delta):
        if proj not in self.user_counts:
            return
        self.user_counts[proj] = max(0, self.user_counts[proj] + delta)
        log.info("用户计数更新: %s = %d", proj, self.user_counts[proj])
        self._adjust_processes(proj)

    def _adjust_processes(self, proj):
        current_users = self.user_counts.get(proj, 0)
        current_processes = len(self.processes.get(proj, []))

        target = math.ceil(current_users / USERS_PER_PROCESS)
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
                log.info("检测到进程退出: %s PID=%d", proj, p.pid)
        self.processes[proj] = alive

    def event_loop(self):
        log.info("进入事件驱动主循环...")
        while self.running:
            try:
                cmd, proj, delta = self.cmd_queue.get(timeout=1.0)
                self.handle_command(cmd, proj, delta)
            except queue.Empty:
                for proj in SPORTS.keys():
                    self._adjust_processes(proj)
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                pass
        log.info("事件循环结束")

    def cleanup(self):
        if not self.running:
            return
        self.running = False
        log.info("正在清理...")

        try:
            self.cmd_queue.put(('stop', None, None))
        except Exception:
            pass

        for proj, procs in self.processes.items():
            for p in procs:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)
                    if p.is_alive():
                        p.kill()

        if hasattr(self, 'django_proc') and self.django_proc.poll() is None:
            self.django_proc.terminate()
            self.django_proc.wait(timeout=3)

        if hasattr(self, 'eval_process') and self.eval_process.is_alive():
            self.eval_process.terminate()
            self.eval_process.join(timeout=2)

        log.info("清理完成")

    def shutdown(self, signum=None, frame=None):
        self.cleanup()

    def run(self):
        log.info("系统启动中...")
        try:
            self.event_loop()
        except KeyboardInterrupt:
            self.shutdown()


if __name__ == "__main__":
    launcher = Launcher()
    launcher.run()
