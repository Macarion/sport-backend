from concurrent.futures import ThreadPoolExecutor
import threading

from api.sport_manager import SportManager
from api.new_situp import SITUP
from api.new_pullup import PULL


class UserManager:
    """
    所有评测系统的管理器
    按用户id进行管理，每个用户同时只能进行一种评测
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):

        if hasattr(self, "_initialized"):
            return

        self._initialized = True

        self.users = {}

        self.users_lock = threading.Lock()

        self.executor = ThreadPoolExecutor(max_workers=8)

    def start_sport_test(self, uid, sport_type):
        self.users_lock.acquire()
        if uid not in self.users:
            try:
                sport = SportManager(uid, sport_type)
                self.users[uid] = sport
                self.executor.submit(self.users[uid].start)
            except ValueError:
                self.users_lock.release()
                return False

        self.users_lock.release()
        return True
    
    def set_output_track(self, uid, output_track):
        self.users_lock.acquire()
        if uid in self.users:
            self.users[uid].set_output_track(output_track)
        self.users_lock.release()

    def push_frame(self, uid, frame):
        self.users_lock.acquire()
        if uid in self.users:
            self.users[uid].push_frame(frame)
        self.users_lock.release()

    def stop_sport_test(self, uid):
        self.users_lock.acquire()
        if uid in self.users:
            self.executor.submit(self.users[uid].stop)
            del self.users[uid]
        self.users_lock.release()
