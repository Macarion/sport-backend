# api/user_manager.py
"""用户会话管理：web 进程内直接 spawn 用户进程（ADR-0001）。

每个体测会话 = 双 mp.Queue（帧入/结果出）+ 4 槽共享内存（同槽就地绘制回传）。
对外协议冻结：start_sport_test/stop_sport_test/push_frame/register_consumer/
unregister_consumer/update_config 的签名与 WS 消息格式保持旧版不变。
"""
import asyncio
import queue
import threading
import time
import multiprocessing as mp

from core.log import get_logger
from core.registry import is_registered
from core.shm import SlotSet, attach_slot
from core.task import SportTask
from worker.run_user_worker import run_user_worker

log = get_logger('UserManager')

IDLE_TIMEOUT = 60           # 会话空闲销毁（与 worker 兜底同口径）
FINISH_JOIN_TIMEOUT = 5     # 发停止信号后等待 worker 退出的上限
POLL_INTERVAL = 0.2         # 结果轮询间隔（秒）


class _Session:
    """单个体测会话的全部资源：进程、队列、共享内存槽。"""

    def __init__(self, uid, sport_type):
        self.uid = uid
        self.sport_type = sport_type
        self.slots = SlotSet(uid)
        self.in_q = mp.Queue()
        self.out_q = mp.Queue()
        self.proc = mp.Process(target=run_user_worker,
                               args=(sport_type, self.in_q, self.out_q),
                               daemon=True)
        self.lock = threading.Lock()   # 保护 free_slots / slot_shapes / frame_id
        self.free_slots = list(self.slots.names)
        self.slot_shapes = {}
        self.frame_id = 0
        self.config = {}
        self.ready = False
        self.finalized = False
        self.stopping_at = None
        self.last_active = time.time()
        self.push_ok = 0
        self.push_drop = 0
        self.res_ok = 0
        self._rate_t = time.time()
        self.pump = None               # asyncio.Task，创建于 daphne 事件循环

    def spawn(self):
        self.proc.start()


class UserManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._sessions: dict[str, _Session] = {}
        self._consumers = {}
        self._loop = None

    # ---------- 会话生命周期 ----------

    def start_sport_test(self, uid, sport_type='situp'):
        uid = str(uid)
        if not is_registered(sport_type):
            log.warning('未注册的体测项目，拒绝启动: uid=%s sport=%s', uid, sport_type)
            return False

        old = self._sessions.pop(uid, None)
        if old is not None:
            log.warning('uid=%s 已有会话，先销毁旧会话', uid)
            self._request_stop(old)
            self._finalize(old)

        session = None
        try:
            session = _Session(uid, sport_type)
            session.spawn()
        except Exception as e:
            log.error('创建用户进程失败: uid=%s err=%s', uid, e)
            if session is not None:
                self._finalize(session)
            return False

        self._sessions[uid] = session
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._ensure_pump, uid)
        log.info('用户开始测试: uid=%s, sport=%s, PID=%d',
                 uid, sport_type, session.proc.pid)
        return True

    def stop_sport_test(self, uid):
        uid = str(uid)
        session = self._sessions.pop(uid, None)
        if session is None:
            return
        self._request_stop(session)
        if session.pump is None or session.pump.done():
            # 没有结果轮询在跑（WS 未注册或 pump 已退出）：就地收尾
            self._finalize(session)
        log.info('用户停止测试: uid=%s', uid)

    def shutdown_all(self):
        """进程退出收口：销毁全部会话。"""
        for uid in list(self._sessions):
            session = self._sessions.pop(uid, None)
            if session is None:
                continue
            self._request_stop(session)
            self._finalize(session)
        log.info('全部用户会话已销毁')

    def _request_stop(self, session):
        session.stopping_at = time.time()
        try:
            task = SportTask(uid=session.uid, frame_id=-1,
                             sport_type=session.sport_type, finished=True)
            session.in_q.put(task.to_bytes())
        except Exception as e:
            log.error('发送停止信号失败: uid=%s err=%s', session.uid, e)

    def _finalize(self, session):
        """等用户进程退出（兜底强杀）并回收共享内存槽。幂等。"""
        if session.finalized:
            return
        session.finalized = True
        try:
            session.proc.join(timeout=3)
            if session.proc.is_alive():
                log.warning('用户进程未退出，强制终止: PID=%d', session.proc.pid)
                session.proc.terminate()
                session.proc.join(timeout=2)
        except Exception as e:
            log.error('回收用户进程失败: %s', e)
        try:
            session.slots.destroy()
        except Exception as e:
            log.error('销毁共享内存槽失败: %s', e)
        for q in (session.in_q, session.out_q):
            try:
                q.close()
            except Exception:
                pass

    # ---------- 消费者（WebSocket）----------

    async def register_consumer(self, consumer):
        uid = str(consumer.uid)
        self._consumers[uid] = consumer
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        self._ensure_pump(uid)
        # 若用户进程在 WS 注册前已就绪，补发 phase（否则前端只能等首帧 data）
        session = self._sessions.get(uid)
        if session is not None and session.ready:
            await consumer.channel_layer.send(consumer.channel_name,
                                              {'type': 'phase', 'phase': 'ready'})

    def unregister_consumer(self, uid):
        # WS 断线是销毁触发器之一：等价于停止测试
        self._consumers.pop(str(uid), None)
        self.stop_sport_test(uid)

    def update_config(self, uid, config_data):
        session = self._sessions.get(str(uid))
        if not session:
            return
        for key, value in config_data.items():
            if key != 'type':
                session.config[key] = value
        log.info('更新配置: uid=%s, config=%s', uid, session.config)

    # ---------- 帧入（WebRTC 收流线程 / executor 调用）----------

    def push_frame(self, uid, frame):
        uid = str(uid)
        session = self._sessions.get(uid)
        if session is None or session.finalized:
            return False

        with session.lock:
            if not session.free_slots:
                session.push_drop += 1   # 4 槽全在途：丢帧（loading 期 / 背压）
                self._rate(session)
                return False
            slot = session.free_slots.pop(0)
            session.slot_shapes[slot] = frame.shape
            frame_id = session.frame_id
            session.frame_id += 1

        try:
            shm, arr = attach_slot(slot, frame.shape)
            arr[:] = frame
            shm.close()
        except Exception as e:
            log.error('写入共享内存槽失败: %s', e)
            with session.lock:
                session.slot_shapes.pop(slot, None)
                session.free_slots.append(slot)
            session.push_drop += 1
            self._rate(session)
            return False

        task = SportTask(uid, frame_id, session.sport_type,
                         slot, frame.shape, session.config)
        session.in_q.put(task.to_bytes())
        session.last_active = time.time()
        session.push_ok += 1
        self._rate(session)
        return True

    @staticmethod
    def _rate(session):
        """每 5 秒汇总一次帧入队/丢弃/结果统计（运维可见性）。"""
        now = time.time()
        if now - session._rate_t >= 5:
            log.info('帧统计 uid=%s: 入队=%d 丢弃=%d 结果=%d 空结果=%d (5s窗口)',
                     session.uid, session.push_ok, session.push_drop, session.res_ok,
                     getattr(session, 'res_none', 0))
            session._rate_t = now

    # ---------- 结果出（daphne 事件循环上的每会话 pump）----------

    def _ensure_pump(self, uid):
        session = self._sessions.get(uid)
        if session is None or session.finalized:
            return
        if session.pump is not None and not session.pump.done():
            return
        session.pump = asyncio.ensure_future(self._pump(session))

    @staticmethod
    def _get_msg(session):
        return session.out_q.get(timeout=POLL_INTERVAL)

    async def _pump(self, session: _Session):
        loop = asyncio.get_event_loop()
        uid = session.uid
        log.info('结果轮询启动: uid=%s', uid)
        try:
            while True:
                if session.stopping_at is not None:
                    if (not session.proc.is_alive()
                            or time.time() - session.stopping_at > FINISH_JOIN_TIMEOUT):
                        break
                try:
                    msg = await loop.run_in_executor(None, self._get_msg, session)
                except queue.Empty:
                    msg = None

                if msg is None:
                    if (session.stopping_at is None
                            and time.time() - session.last_active > IDLE_TIMEOUT):
                        log.info('会话空闲 %d 秒，自动销毁: uid=%s', IDLE_TIMEOUT, uid)
                        self._sessions.pop(uid, None)
                        self._request_stop(session)
                    continue

                mtype = msg.get('type')
                if mtype == 'phase':
                    session.ready = True
                    log.info('用户进程就绪: uid=%s', uid)
                    consumer = self._consumers.get(uid)
                    if consumer is not None:
                        await consumer.channel_layer.send(
                            consumer.channel_name,
                            {'type': 'phase', 'phase': msg.get('phase', 'ready')})
                elif mtype == 'result':
                    await self._on_result(session, msg, loop)
                elif mtype == 'finished':
                    break
                elif mtype == 'error':
                    log.error('用户进程报错: uid=%s, msg=%s', uid, msg.get('message'))
                    consumer = self._consumers.get(uid)
                    if consumer is not None:
                        await consumer.channel_layer.send(
                            consumer.channel_name,
                            {'type': 'error', 'message': msg.get('message', '')})
                    break
        finally:
            self._sessions.pop(uid, None)
            consumer = self._consumers.pop(uid, None)
            await loop.run_in_executor(None, self._finalize, session)
            if consumer is not None:
                try:
                    await consumer.channel_layer.send(consumer.channel_name,
                                                      {'type': 'test_end'})
                except Exception as e:
                    log.error('发送 test_end 失败: uid=%s, err=%s', uid, e)
            log.info('会话结束: uid=%s', uid)

    async def _on_result(self, session: _Session, msg, loop):
        session.res_ok += 1
        self._rate(session)
        slot = msg.get('slot')
        if slot:
            with session.lock:
                shape = session.slot_shapes.pop(slot, None)
                if slot not in session.free_slots:
                    session.free_slots.append(slot)   # 槽归还（先于投递，尽快复用）
            frame = await loop.run_in_executor(None, self._read_slot, slot, shape)
            consumer = self._consumers.get(session.uid)
            if frame is not None and consumer is not None:
                consumer.output_track.push_frame(frame)

        data = msg.get('data')
        if data is None:
            session.res_none = getattr(session, 'res_none', 0) + 1
        consumer = self._consumers.get(session.uid)
        if data is not None and consumer is not None:
            await consumer.channel_layer.send(consumer.channel_name,
                                              {'type': 'eval_result', 'data': data})

    @staticmethod
    def _read_slot(slot, shape):
        if shape is None:
            return None
        shm, arr = attach_slot(slot, shape)
        try:
            return arr.copy()
        finally:
            shm.close()


user_manager = UserManager()
