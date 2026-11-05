# core/face_service.py
"""人脸服务进程（ADR-0004/0005）：web 启动即异步预载模型，单任务识别，结果直回请求方。

请求：worker 的 face_client 经共享请求队列投递
      {'type':'recognize','student_id','crop','frame_id','timestamp','reply_to'}
应答：reply_to.send({...,'timestamp'}) 直回所属 worker 的专用结果管道
      （写端随请求捎带，读端由 face_client 在调用方线程 poll 直读，
      无转发线程、不经 in_q，全程不经 web）。
      未注册学生先查库直接放行（18.b 过渡，不进识别），已注册才调 recognize。
"""
import time
import multiprocessing as mp

from core.config import FACE_DB_PATH, FACE_MODEL_DIR
from core.log import get_logger

log = get_logger('FaceService')

_request_q = None


def get_request_q():
    """web 进程内的共享请求队列；main.py 与 user_manager 各取一次，同一实例。"""
    global _request_q
    if _request_q is None:
        _request_q = mp.Queue()
    return _request_q


def spawn_face_service():
    """异步启动人脸服务并立即返回（166MB 模型预载在子进程内完成）。"""
    proc = mp.Process(target=run_face_service, args=(get_request_q(),),
                      name='FaceService', daemon=True)
    proc.start()
    log.info('人脸服务已启动: PID=%d（预载完成前的请求在队列内等待）', proc.pid)
    return proc


def run_face_service(request_q):
    mp.current_process().name = 'FaceService'
    from face_sdk import init_sdk, is_registered, recognize

    FACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)   # data/ 落盘处
    t0 = time.time()
    try:
        init_sdk(model_dir=FACE_MODEL_DIR, db_path=FACE_DB_PATH)
    except Exception as e:
        log.error('人脸服务初始化失败（此后门禁请求将全部超时重试）: %s', e)
        return
    log.info('人脸识别模型预载完成 耗时%.1fs', time.time() - t0)

    while True:
        try:
            msg = request_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.get('type') != 'recognize':
            continue
        reply = msg.get('reply_to')
        if reply is None:
            continue
        student_id = msg.get('student_id')
        out = {'type': 'face_result', 'student_id': student_id,
               'frame_id': msg.get('frame_id'), 'timestamp': msg.get('timestamp'),
               'ok': False, 'conf': 0.0}
        try:
            if not is_registered(student_id):
                out['registered'] = False   # 18.b 未注册放行：不进识别
                out['ok'] = True
            else:
                ok, conf, _ = recognize(student_id, msg['crop'])
                out['ok'] = bool(ok)
                out['conf'] = float(conf)
                out['registered'] = True
        except Exception as e:
            log.error('识别失败: student_id=%s err=%s', student_id, e)
        try:
            reply.send(out)
        except Exception as e:
            log.warning('应答回发失败: student_id=%s err=%s', student_id, e)
        finally:
            try:
                reply.close()
            except Exception:
                pass
