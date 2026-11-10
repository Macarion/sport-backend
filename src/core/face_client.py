# core/face_client.py
"""人脸识别客户端类（ADR-0005）。

worker 启动时实例化（门禁关闭即 face_request_q 为 None 时为 None），经
BaseSport.set_face_client 注入 sport，每会话由 worker 调 reset()。人脸核验闸门
face_recognition 由 Evaluator 在坐姿合格后（流程阶段一）调用；结论不发 phase——经
status() 随 Evaluator 返回值（data.face）承载给前端。

单任务限流：send 时已有在途请求（未超时）→ 返回 False（本帧丢弃识别资格，
不缓存不排队），应答到达或超时后由下一帧重新触发。

通路：请求经共享 request_q 捎带实例专用结果管道的写端；应答由 poll() 在调用方
线程 poll(0) 非阻塞直读——无转发线程、不经 worker 的 in_q。ADR-0004 三不变量
保留：不经 web、直回请求方、单任务无排队。

陈旧应答过滤：应答回带学号 + 请求毫秒时间戳，poll 只保留与最近一次 send
完全对应的结果，其余丢弃。
"""
import time

import multiprocessing as mp

from core.config import FACE_GATE_TIMEOUT
from core.log import get_logger

log = get_logger('FaceClient')


class FaceClient:
    """人脸识别客户端：持有专用结果管道、在途状态与门禁放行状态。"""

    def __init__(self, request_q, timeout=FACE_GATE_TIMEOUT):
        self._request_q = request_q     # 人脸服务共享请求队列（None=禁用）
        self._timeout = timeout
        self._student_id = None         # 最近一次 send 的学号（应答过滤键）
        self._timestamp = 0             # 最近一次 send 的毫秒时间戳（应答过滤键）
        self._last_request_timestamp = 0  # survives reset: same-ms target/session changes cannot collide
        self._in_flight = False
        self._t0 = 0.0
        self._timed_out = False         # 超时已发生、待 poll 上报一次
        self._face_ok = False           # 门禁已放行（会话内）
        self._gate_opened = False       # 门禁已开（status 开始出值）
        self._last = None               # 最近一次核验结论（status 快照）
        self._fail_count = 0            # 会话内失败/超时累计（前端第3次起才提示）
        if request_q is not None:
            # 每实例一条持久管道：写端随每请求 pickle 捎带，读端长期持有防 EOF
            self._recv_end, self._send_end = mp.Pipe(duplex=False)
        else:
            self._recv_end = self._send_end = None

    def reset(self):
        """会话开始清全部状态并排空管道（worker 每会话调用）。"""
        self._student_id, self._timestamp = None, 0
        self._in_flight, self._t0, self._timed_out = False, 0.0, False
        self._face_ok, self._gate_opened, self._last = False, False, None
        self._fail_count = 0
        if self._recv_end is not None:
            while self._recv_end.poll(0):
                try:
                    self._recv_end.recv()
                except (EOFError, OSError):
                    break

    def send(self, student_id, crop) -> bool:
        """投递识别请求；门禁关闭或在途未超时 → False（本帧丢弃，不排队）。"""
        if self._request_q is None or self._in_flight:
            return False
        self._student_id = student_id
        self._timestamp = max(int(time.time() * 1000), self._last_request_timestamp + 1)
        self._last_request_timestamp = self._timestamp
        self._request_q.put({'type': 'recognize', 'student_id': student_id,
                             'crop': crop, 'timestamp': self._timestamp,
                             'reply_to': self._send_end})
        self._in_flight, self._t0 = True, time.time()
        self._timed_out = False
        return True

    def poll(self):
        """取识别结论（调用方每帧先 poll 后 send）。

        在途：先非阻塞读——命中 学号+时间戳 对应的结果即返回并清在途；读空且超过
        timeout → 返回 {'ok': False, 'timeout': True} 一次并放行重投。
        不在途：返回 None（不读管道；迟到应答留待下一次 send 后按戳过滤丢弃）。
        """
        if self._in_flight:
            out = None
            while self._recv_end.poll(0):
                try:
                    msg = self._recv_end.recv()
                except (EOFError, OSError) as e:
                    log.warning('人脸应答接收失败: %s', e)
                    break
                # 只保留与最近一次 send 完全对应的结果，其余（迟到/陈旧）丢弃
                if (msg.get('student_id') == self._student_id
                        and msg.get('timestamp') == self._timestamp):
                    out = msg
            if out is not None:
                self._in_flight, self._t0 = False, 0.0
                self._timestamp = 0
                return out
            if time.time() - self._t0 > self._timeout:
                log.warning('识别结果超时（>%.0fs），清除在途标记', self._timeout)
                self._in_flight, self._t0 = False, 0.0
                self._timed_out = True
        if self._timed_out:
            self._timed_out = False
            return {'type': 'face_result', 'ok': False, 'timeout': True}
        return None

    def face_recognition(self, student_id, crop) -> bool:
        """人脸识别闸门（Evaluator 阶段一坐姿合格后调用）：True=人脸已通过。

        每帧先 poll 消费应答/超时（按学号+时间戳对齐），未放行则投递新识别请求。
        不发任何 phase：结论经 status() 随评测返回值承载。
        """
        self._gate_opened = True
        res = self.poll()
        if res is not None:
            if res.get('timeout'):
                self._fail_count += 1
                self._last = {'state': 'timeout', 'ok': False,
                              'conf': 0.0, 'registered': True}
            elif res.get('registered') is False:
                self._face_ok = True                    # 18.b 未注册放行（不进识别）
                self._last = {'state': 'passed', 'ok': True,
                              'conf': 0.0, 'registered': False}
            elif res.get('ok'):
                self._face_ok = True
                self._last = {'state': 'passed', 'ok': True,
                              'conf': float(res.get('conf') or 0.0),
                              'registered': True}
            else:
                self._fail_count += 1
                self._last = {'state': 'failed', 'ok': False,
                              'conf': float(res.get('conf') or 0.0),
                              'registered': bool(res.get('registered', True))}
        if self._face_ok:
            return True
        if crop is not None:
            self.send(student_id, crop)
        return self._face_ok

    def status(self):
        """人脸状态快照（Evaluator 返回值 data.face 承载）：
        未进门禁 → None；核验中 → checking；结论后 passed/failed/timeout
        持续到下次结论；fail_count 为会话内失败/超时累计（前端第3次起才提示）。
        """
        if not self._gate_opened:
            return None
        snap = self._last or {'state': 'checking', 'ok': None,
                              'conf': 0.0, 'registered': None}
        return {**snap, 'fail_count': self._fail_count}
