# core/task.py
from contextlib import contextmanager

from .shm import attach_slot


class SportTask:
    """进程间传递任务信息的类（唯一定义，web 与用户进程共用）。

    序列化铁律：只经 to_dict() 白名单字段过队列；_shm/_frame 是进程内
    attach 句柄，绝不进消息——否则整块视频帧会随管道传输，共享内存失去意义。
    """

    def __init__(self, uid, frame_id, sport_type, shm_name=None, frame_shape=None,
                 config=None, data=None):
        self.uid = uid
        self.frame_id = frame_id
        self.sport_type = sport_type
        self.shm_name = shm_name
        self.frame_shape = frame_shape
        self.config = config
        self.data = data

        self._shm = None
        self._frame = None

    def get_frame(self):
        """获取存储在共享内存中的视频帧地址"""
        return self._frame

    def set_frame(self, frame):
        """直接注入已就位的帧（本地链式处理用，不经过共享内存 attach）"""
        self._frame = frame

    def _init_frame(self):
        """初始化共享内存视频帧引用（直接在返回地址上修改即为修改共享内存）。"""
        if self.shm_name is None:
            return None
        self._shm, self._frame = attach_slot(self.shm_name, self.frame_shape)
        return self._frame

    def _close_frame(self):
        """关闭共享内存视频帧引用，处理完一帧后调用。"""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        self._frame = None

    def get_uid(self):
        return self.uid

    def get_frame_id(self):
        return self.frame_id

    def get_sport_type(self):
        return self.sport_type

    def get_shm_name(self):
        return self.shm_name

    def get_frame_shape(self):
        return self.frame_shape

    def set_config(self, config):
        """设置评测配置"""
        self.config = config

    def get_config(self):
        """获取评测配置，初始配置由前端传入"""
        return self.config

    def set_data(self, data):
        """设置程序间传输的自定义数据，最后一个进程设置的数据将被发送到前端"""
        self.data = data

    def get_data(self):
        """获取程序间传输的自定义数据"""
        return self.data

    def __reduce__(self):
        """禁止整对象 pickle：直接 put(task) 会把 _frame/_shm 拖过管道。"""
        raise TypeError('SportTask 必须经 to_dict() 白名单序列化后入队，禁止直接 put(task)')

    @contextmanager
    def link_frame(self):
        """上下文管理：自动初始化和关闭共享内存引用。"""
        try:
            self._init_frame()
            yield self
        finally:
            self._close_frame()

    def to_dict(self):
        """白名单字段序列化（in_q 信封的任务体）；_shm/_frame 永不入列。"""
        return {
            'uid': self.uid,
            'frame_id': self.frame_id,
            'sport_type': self.sport_type,
            'shm_name': self.shm_name,
            'frame_shape': self.frame_shape,
            'config': self.config,
            'data': self.data,
        }

    @classmethod
    def from_dict(cls, data):
        """从信封任务体反序列化；_frame 由调用方经 link_frame() 按名 attach。"""
        return cls(
            uid=data['uid'],
            frame_id=data['frame_id'],
            sport_type=data['sport_type'],
            shm_name=data.get('shm_name'),
            frame_shape=data.get('frame_shape'),
            config=data.get('config'),
            data=data.get('data'),
        )
