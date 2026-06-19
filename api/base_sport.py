class BaseSport:

    """
    基础运动类，定义了运动类的基本接口和功能框架
    这是一个抽象基类，具体的运动类需要继承并实现其中的方法
    """
    def __init__(self, uid = None):
        """
        初始化方法
        :param uid: 用户唯一标识符，暂无用处
        """
        ...

    def start(self): 
        """
        开始运动的方法
        这是一个抽象方法，需要在子类中实现具体逻辑
        """
        raise NotImplementedError

    def stop(self): 
        """
        停止运动的方法
        这是一个抽象方法，需要在子类中实现具体逻辑
        """
        raise NotImplementedError

    def update(self, frame, frame_idx): 
        """
        更新每一帧

        返回值:
            result: 字典，包含运动结果
            painting: numpy 数组，可视化图像 
        """

        raise NotImplementedError
