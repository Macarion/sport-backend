import numpy as np

class KalmanFilter2D:
    """
    2D卡尔曼滤波器，用于过滤坐标数据
    适用于平滑x、y坐标的噪声
    """
    def __init__(self, process_variance=1e-4, measurement_variance=1e-1, initial_estimate_error=1.0):
        """
        初始化卡尔曼滤波器

        参数:
        process_variance: 过程噪声方差，表示系统状态变化的不确定性
        measurement_variance: 测量噪声方差，表示测量值的不确定性
        initial_estimate_error: 初始估计误差
        """
        # 状态向量 [x, y, vx, vy] - 位置和速度
        self.state_dim = 4
        self.measurement_dim = 2  # 只测量位置 [x, y]

        # 状态转移矩阵 (4x4)
        # 假设匀速运动模型
        dt = 1.0  # 时间步长
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 观测矩阵 (2x4) - 只观测位置
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 过程噪声协方差矩阵 (4x4)
        self.Q = np.eye(self.state_dim) * process_variance

        # 测量噪声协方差矩阵 (2x2)
        self.R = np.eye(self.measurement_dim) * measurement_variance

        # 初始状态协方差矩阵 (4x4)
        self.P = np.eye(self.state_dim) * initial_estimate_error

        # 初始状态向量 [x, y, vx, vy]
        self.x = np.zeros(self.state_dim)

        # 标志位：是否已经初始化
        self.initialized = False

    def initialize(self, initial_measurement):
        """
        使用第一个测量值初始化滤波器

        参数:
        initial_measurement: 初始测量值 [x, y]
        """
        if len(initial_measurement) >= 2:
            self.x[:2] = np.array(initial_measurement[:2])  # 设置初始位置
            self.x[2:] = np.zeros(2)  # 初始速度为0
            self.initialized = True

    def predict(self):
        """
        预测步骤：根据当前状态预测下一状态
        """
        if not self.initialized:
            return None

        # 状态预测
        self.x = np.dot(self.A, self.x)

        # 协方差预测
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x.copy()

    def update(self, measurement):
        """
        更新步骤：使用新的测量值校正预测状态

        参数:
        measurement: 测量值 [x, y]

        返回:
        滤波后的状态向量 [x, y, vx, vy]
        """
        if not self.initialized:
            self.initialize(measurement)
            return self.x.copy()

        if len(measurement) < 2:
            return self.x.copy()

        z = np.array(measurement[:2])  # 测量值

        # 计算卡尔曼增益
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 状态更新
        y = z - np.dot(self.H, self.x)  # 测量残差
        self.x = self.x + np.dot(K, y)

        # 协方差更新
        I = np.eye(self.state_dim)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.x.copy()

    def filter(self, measurement):
        """
        执行完整的滤波步骤：预测 + 更新

        参数:
        measurement: 测量值 [x, y]

        返回:
        滤波后的位置 [x, y]
        """
        self.predict()
        filtered_state = self.update(measurement)
        return filtered_state[:2]  # 只返回位置

    def reset(self):
        """
        重置滤波器状态
        """
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1.0
        self.initialized = False


class MultiPointKalmanFilter:
    """
    多点卡尔曼滤波器，可以同时过滤多个关键点的坐标
    """
    def __init__(self, process_variance=1e-4, measurement_variance=1e-1):
        """
        初始化多点卡尔曼滤波器

        参数:
        process_variance: 过程噪声方差
        measurement_variance: 测量噪声方差
        """
        self.filters = {}  # 为每个关键点维护一个独立的卡尔曼滤波器
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def filter_point(self, point_name, measurement):
        """
        过滤单个关键点的坐标

        参数:
        point_name: 关键点名称 (如 'nose', 'left_shoulder')
        measurement: 测量值 [x, y]

        返回:
        滤波后的坐标 [x, y]
        """
        if point_name not in self.filters:
            # 为新关键点创建滤波器
            self.filters[point_name] = KalmanFilter2D(
                process_variance=self.process_variance,
                measurement_variance=self.measurement_variance
            )

        # 执行滤波
        filtered_position = self.filters[point_name].filter(measurement)
        return filtered_position

    def filter_keypoints(self, keypoints_dict):
        """
        过滤多个关键点的坐标

        参数:
        keypoints_dict: 关键点字典 {'point_name': {'x': x, 'y': y, ...}, ...}

        返回:
        滤波后的关键点字典
        """
        filtered_keypoints = {}

        for point_name, point_data in keypoints_dict.items():
            if 'x' in point_data and 'y' in point_data:
                measurement = [point_data['x'], point_data['y']]
                filtered_pos = self.filter_point(point_name, measurement)

                # 复制原始数据并更新坐标
                filtered_keypoints[point_name] = point_data.copy()
                filtered_keypoints[point_name]['x'] = filtered_pos[0]
                filtered_keypoints[point_name]['y'] = filtered_pos[1]

        return filtered_keypoints

    def reset_all(self):
        """
        重置所有滤波器
        """
        for filter_obj in self.filters.values():
            filter_obj.reset()
        self.filters.clear()

    def reset_point(self, point_name):
        """
        重置特定关键点的滤波器

        参数:
        point_name: 关键点名称
        """
        if point_name in self.filters:
            self.filters[point_name].reset()


# 便捷函数
def create_coordinate_filter(process_variance=1e-4, measurement_variance=1e-1):
    """
    创建坐标滤波器

    参数:
    process_variance: 过程噪声方差 (默认1e-4)
    measurement_variance: 测量噪声方差 (默认1e-1)

    返回:
    MultiPointKalmanFilter实例
    """
    return MultiPointKalmanFilter(
        process_variance=process_variance,
        measurement_variance=measurement_variance
    )


# 测试函数
if __name__ == "__main__":
    # 测试卡尔曼滤波器
    print("测试卡尔曼滤波器...")

    # 创建滤波器
    filter_obj = create_coordinate_filter()

    # 模拟有噪声的坐标数据
    import random
    true_positions = [(100 + i*2, 200 + i*1) for i in range(10)]  # 真实轨迹

    noisy_positions = []
    for true_x, true_y in true_positions:
        # 添加噪声
        noisy_x = true_x + random.gauss(0, 3)  # 标准差3的噪声
        noisy_y = true_y + random.gauss(0, 3)
        noisy_positions.append((noisy_x, noisy_y))

    print("原始有噪声数据:")
    for i, (x, y) in enumerate(noisy_positions[:5]):
        print(f"  点{i}: ({x:.2f}, {y:.2f})")

    # 应用滤波
    filtered_positions = []
    for x, y in noisy_positions:
        filtered = filter_obj.filter_point('test_point', [x, y])
        filtered_positions.append(filtered)

    print("\n滤波后数据:")
    for i, (x, y) in enumerate(filtered_positions[:5]):
        print(f"  点{i}: ({x:.2f}, {y:.2f})")

    print("\n✅ 卡尔曼滤波器测试完成")
