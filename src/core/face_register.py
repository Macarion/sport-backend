# core/face_register.py
"""人脸注册工具：学号 + 一张 BGR ndarray → data/faces.db（与人脸服务同库同模型）。

库内调用：from core.face_register import register_face
命令行（src/ 为当前目录，或直跑脚本）：
    python -m core.face_register <学号> <图片路径>
    python core/face_register.py <学号> <图片路径>
同 学号 重复注册会覆盖旧特征（sdk 内先删后建）。
"""
import sys
from pathlib import Path

if __package__ in (None, ''):          # 直跑脚本时把 src 补进 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.config import FACE_DB_PATH, FACE_MODEL_DIR

_inited = False                        # 每进程只 init_sdk 一次


def register_face(student_id: str, image: np.ndarray) -> bool:
    """注册一张人脸照片到 faces.db。

    Args:
        student_id: 学号
        image: BGR 格式 ndarray（cv2 读出的色彩序）

    Returns:
        True 注册成功；False 图中未检测到人脸（注册失败）。
    """
    global _inited
    from face_sdk import init_sdk, register
    if not _inited:
        FACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        init_sdk(model_dir=FACE_MODEL_DIR, db_path=FACE_DB_PATH)
        _inited = True
    return bool(register(str(student_id), [image]))


def imread(path: str):
    """读图，支持中文路径（README 约定：fromfile + imdecode；cv2.imread 不行）。"""
    import cv2
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description='注册人脸到 faces.db（学号 + 图片）')
    parser.add_argument('student_id', help='学号')
    parser.add_argument('image', help='图片文件路径（支持中文路径）')
    args = parser.parse_args(argv)

    img = imread(args.image)
    if img is None:
        print(f'FAIL 图片读取失败: {args.image}')
        return 1
    if register_face(args.student_id, img):
        print(f'PASS 学号 {args.student_id} 注册成功 → {FACE_DB_PATH}')
        return 0
    print(f'FAIL 图中未检测到人脸: {args.image}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
