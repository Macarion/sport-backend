import pickle


class SportTask:

    """
    进程间传递任务信息的类
    """

    def __init__(self, uid, frame_id, sport_type, shm_name, config=None, data=None, finished=False):

        self.uid = uid
        self.frame_id = frame_id
        self.sport_type = sport_type
        self.shm_name = shm_name
        self.config = config
        self.data = data
        self.finished = finished

        self._frame = None
    
    def get_frame(self):
        """
        获取存储在共享内存中的视频帧地址
        """
        return self._frame
    
    def set_frame(self, frame):
        """
        设置共享内存视频帧地址 \\
        注意不是修改共享内存中的视频帧，若想修改共享内存的视频帧，请直接在get_frame()返回的地址上修改 \\
        ⚠ 该函数由进程间管理逻辑自动执行
        """
        self._frame = frame
    
    def clear_frame(self):
        """
        清除共享内存视频帧地址，在处理完一帧后调用，防止在队列中传输完整的视频帧 \\
        ⚠ 该函数由进程间管理逻辑自动执行
        """
        self._frame = None
    
    def get_uid(self):
        return self.uid
    
    def get_frame_id(self):
        return self.frame_id

    def get_sport_type(self):
        return self.sport_type

    def get_shm_name(self):
        return self.shm_name

    def set_config(self, config):
        """
        设置评测配置
        """
        self.config = config

    def get_config(self):
        """
        获取评测配置，初始的配置由前端传入
        """
        return self.config

    def set_data(self, data):
        """
        设置程序间传输的自定义数据，最后一个进程设置的数据将被发送到前端 \\
        ⚠ 该函数由进程间管理逻辑自动执行
        """
        self.data = data

    def get_data(self):
        """
        获取程序间传输的自定义数据，最后一个进程设置的数据将被发送到前端
        """
        return self.data
    
    def set_finished(self, finished):
        self.finished = finished

    def is_finished(self):
        return self.finished
    
    def to_bytes(self):
        """由于Web服务端使用的SportTask和推理、评测进程使用的SportTask不由同一个文件定义，因此需要手动序列化为字节"""
        data = {
            'uid': self.uid,
            'frame_id': self.frame_id,
            'sport_type': self.sport_type,
            'shm_name': self.shm_name,
            'config': self.config,
            'data': self.data,
            'finished': self.finished,
        }
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, byte_data):
        """从字节反序列化"""
        data = pickle.loads(byte_data)
        return cls(
            uid=data['uid'],
            frame_id=data['frame_id'],
            sport_type=data['sport_type'],
            shm_name=data['shm_name'],
            config=data.get('config'),
            data=data.get('data'),
            finished=data.get('finished', False),
        )