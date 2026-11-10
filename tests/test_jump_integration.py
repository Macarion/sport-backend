import asyncio
import multiprocessing as mp
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from core.face_client import FaceClient
from core.shm import SlotSet, attach_slot
from core.task import SportTask
from sports.jump import model_runtime
from sports.jump.session import FastReadyWebSession, FastReadyConfig
from sports.jump.sport import SportJump


class FakeSport:
    def __init__(self): self.loads = 0; self.sessions = 0
    def set_face_client(self, client): pass
    def load(self): self.loads += 1
    def reset(self): self.sessions += 1
    def process(self, frame, config, frame_id, uid=0, timestamp_ms=None):
        frame[:] += 1
        if config.get('crash'): raise RuntimeError('deliberate failure')
        return {'uid': uid, 'loads': self.loads, 'sessions': self.sessions, 'time': timestamp_ms}
    def end_session(self): pass
    def close(self): pass


def fake_worker(in_q, out_q, idle=60):
    import worker.run_user_worker as entry
    entry.load_sport = lambda _: FakeSport
    entry.IDLE_TIMEOUT = idle
    entry.run_user_worker('jump', in_q, out_q)


def manager_fake_worker(sport_type, in_q, out_q, face_request_q=None):
    fake_worker(in_q, out_q)


class WorkerContracts(unittest.TestCase):
    def start_worker(self, idle=60):
        context = mp.get_context('spawn')
        iq, oq = context.Queue(), context.Queue()
        process = context.Process(target=fake_worker, args=(iq, oq, idle))
        process.start()
        def cleanup():
            if process.is_alive(): process.terminate()
            process.join(5)
            iq.close(); oq.close()
        self.addCleanup(cleanup)
        return process, iq, oq

    def test_ten_sessions_share_one_model_and_return_drawn_slots(self):
        process, iq, oq = self.start_worker()
        slots = SlotSet('worker_test'); self.addCleanup(slots.destroy)
        shm, frame = attach_slot(slots.names[0], (4, 5, 3)); self.addCleanup(shm.close)
        for index in range(10):
            iq.put({'type': 'start'})
            self.assertEqual(oq.get(timeout=10), {'type': 'loaded', 'value': True})
            frame[:] = index
            task = SportTask(str(index), 0, 'jump', slots.names[0], frame.shape, {}, timestamp_ms=123 + index)
            iq.put({'type': 'task', 'task': task.to_dict()})
            result = oq.get(timeout=10)
            self.assertEqual(result['data'], {'uid': str(index), 'loads': 1, 'sessions': index + 1, 'time': 123 + index})
            self.assertTrue(np.all(frame == index + 1))
            iq.put({'type': 'stop'})
            self.assertEqual(oq.get(timeout=10), {'type': 'finished'})
            self.assertTrue(process.is_alive())
        iq.put({'type': 'stop'}); process.join(5)
        self.assertFalse(process.is_alive())

    def test_idle_worker_exits(self):
        process, iq, oq = self.start_worker(idle=0.2)
        iq.put({'type': 'start'}); self.assertEqual(oq.get(timeout=10)['type'], 'loaded')
        iq.put({'type': 'stop'}); self.assertEqual(oq.get(timeout=5)['type'], 'finished')
        process.join(5); self.assertFalse(process.is_alive())

    def test_worker_crash_reports_error_not_finished(self):
        process, iq, oq = self.start_worker()
        slots = SlotSet('crash_test'); self.addCleanup(slots.destroy)
        iq.put({'type': 'start'}); oq.get(timeout=10)
        task = SportTask('a', 0, 'jump', slots.names[0], (2, 2, 3), {'crash': True})
        iq.put({'type': 'task', 'task': task.to_dict()})
        self.assertEqual(oq.get(timeout=5)['type'], 'error')
        process.join(5); self.assertFalse(process.is_alive())


class SessionContracts(unittest.TestCase):
    def session(self, face=None):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        session = FastReadyWebSession('student', Path(directory.name),
                                      config=FastReadyConfig(auto_ready_frames=1), face_client=face)
        self.addCleanup(session.stop)
        return session

    def test_buffer_owns_frame_before_shm_overwrite(self):
        session = self.session()
        frame = np.zeros((5, 5, 3), np.uint8)
        captured = []
        with patch.object(session._raw_encode_queue, 'put_nowait', side_effect=captured.append):
            session.ingest_stream_frame(frame, 0, 123)
        frame[:] = 255
        self.assertEqual(captured[0][0].max(), 0)
        self.assertFalse(np.shares_memory(frame, captured[0][0]))

    def test_stop_cancels_late_result_and_joins_threads(self):
        session = self.session()
        session.state = 'PROCESSING'
        def measurement():
            session._cancel.wait(2)
            return {'score_cm': 123}
        session._run_measurement = measurement
        session.processing_thread = threading.Thread(target=session._processing_worker)
        session.processing_thread.start()
        session.stop()
        self.assertEqual(session.state, 'STOPPED')
        self.assertIsNone(session.score_cm)
        self.assertFalse(any(t.is_alive() for t in (session.processing_thread, session._raw_encoder, session._diag_writer)))
        session.stop()  # idempotent

    def test_cleanup_timeout_is_failure(self):
        session = self.session(); release = threading.Event()
        thread = threading.Thread(target=lambda: release.wait(5))
        session.processing_thread = thread; thread.start()
        session._close_deadline = time.monotonic() + 0.05
        try:
            with self.assertRaisesRegex(RuntimeError, 'cannot be reused'): session.stop()
        finally:
            release.set(); thread.join(2)
            session._raw_encoder.join(1); session._diag_writer.join(1)

    def test_face_gates_arming_and_target_loss_resets_identity(self):
        face = Mock(); face.face_recognition.side_effect = [False, True]
        face.status.return_value = {'state': 'checking'}
        session = self.session(face); session.state = 'WAITING_READY'
        person = SimpleNamespace(bbox=[1, 1, 8, 9], bottom_center_world=[0, 0], y2=9, confidence=.9)
        realtime = SimpleNamespace(detect_people=lambda *a: [person],
                                  match_locked_person=lambda *a, **k: person)
        session._select_fast_ready_person = lambda people: person if people else None
        frame = np.zeros((10, 10, 3), np.uint8)
        with patch('sports.jump.session.get_algorithms', return_value=SimpleNamespace(realtime_core=realtime)), \
             patch('sports.jump.session.get_model_bundle', return_value=SimpleNamespace(yolo_model=None)):
            self.assertEqual(session._handle_waiting_ready(frame, 1, 1000)['state'], 'WAITING_READY')
            crop = face.face_recognition.call_args.args[1]
            self.assertFalse(np.shares_memory(frame, crop))
            realtime.detect_people = lambda *a: []
            session._handle_waiting_ready(frame, 2, 1100)
            face.reset.assert_called_once()
            realtime.detect_people = lambda *a: [person]
            self.assertEqual(session._handle_waiting_ready(frame, 3, 1200)['state'], 'ARMED')

    def test_face_target_switch_clears_pending_identity(self):
        face = Mock(); face.face_recognition.return_value = False
        face.status.return_value = {'state':'checking'}
        session = self.session(face); session.state = 'WAITING_READY'
        person = SimpleNamespace(bbox=[0,0,4,9], bottom_center_world=[0,0], y2=9, confidence=.9)
        realtime = SimpleNamespace(detect_people=lambda *a:[person], bbox_iou=lambda *a:0,
                                  match_locked_person=lambda *a, **k:person)
        session._select_fast_ready_person=lambda people:people[0]
        with patch('sports.jump.session.get_algorithms', return_value=SimpleNamespace(realtime_core=realtime)), \
             patch('sports.jump.session.get_model_bundle', return_value=SimpleNamespace(yolo_model=None)):
            frame=np.zeros((10,10,3),np.uint8)
            session._handle_waiting_ready(frame,0,1000)
            person=SimpleNamespace(bbox=[6,0,10,9], bottom_center_world=[0,0], y2=9, confidence=.9)
            session._handle_waiting_ready(frame,1,1100)
            face.reset.assert_called_once()
            self.assertEqual(session.state,'WAITING_READY')

    def test_sport_rejects_wrong_dimensions_and_user(self):
        sport = SportJump(); sport._models = object()
        with self.assertRaises(ValueError): sport.process(np.zeros((721, 1280, 3), np.uint8), {}, 0)
        sport.session = SimpleNamespace(uid='a')
        with self.assertRaises(RuntimeError): sport.process(np.zeros((2, 2, 3), np.uint8), {}, 0, 'b')


class FaceContracts(unittest.TestCase):
    def client(self):
        client = FaceClient(Mock(), timeout=.01)
        self.addCleanup(client._recv_end.close); self.addCleanup(client._send_end.close)
        return client

    def test_registered_unregistered_failed_and_late_replies(self):
        for registered, ok in [(True, True), (False, False), (True, False)]:
            client = self.client()
            with patch('core.face_client.time.time', return_value=10):
                self.assertFalse(client.face_recognition('a', np.zeros((2, 2, 3))))
                client._send_end.send({'student_id': 'wrong', 'timestamp': 10000, 'ok': True})
                client._send_end.send({'student_id': 'a', 'timestamp': 10000, 'registered': registered, 'ok': ok})
                self.assertEqual(client.face_recognition('a', None), ok or not registered)
                self.assertEqual(client.status()['registered'], registered)
            client.reset(); self.assertIsNone(client.status())

    def test_same_millisecond_reset_rejects_old_target_reply(self):
        client = self.client()
        with patch('core.face_client.time.time', return_value=10):
            client.send('same-student', np.zeros((2, 2, 3)))
            old_stamp = client._timestamp
            client.reset()
            client.send('same-student', np.zeros((2, 2, 3)))
            self.assertGreater(client._timestamp, old_stamp)
            client._send_end.send({'student_id': 'same-student', 'timestamp': old_stamp, 'ok': True})
            self.assertFalse(client.face_recognition('same-student', None))
            self.assertNotEqual(client.status()['state'], 'passed')

    def test_timeout_accumulates_and_retries(self):
        client = self.client()
        for index in range(3):
            with patch('core.face_client.time.time', return_value=index * 2 + 1):
                client.face_recognition('a', np.zeros((2, 2, 3)))
            with patch('core.face_client.time.time', return_value=index * 2 + 2):
                self.assertFalse(client.face_recognition('a', None))
        self.assertEqual(client.status()['state'], 'timeout')
        self.assertEqual(client.status()['fail_count'], 3)


class SharedMemoryContracts(unittest.TestCase):
    def test_output_is_copied_before_slot_is_available(self):
        from api.user_manager import UserManager, _Session
        manager = object.__new__(UserManager); manager.__init__()
        session = _Session('slot_test', 'jump', proc=Mock(), in_q=queue.Queue(), out_q=queue.Queue())
        self.addCleanup(session.slots.destroy)
        slot = session.free_slots.pop(0); session.slot_shapes[slot] = (2, 2, 3)
        original = manager._read_slot
        def read(name, shape):
            self.assertNotIn(slot, session.free_slots)
            return original(name, shape)
        manager._read_slot = read
        async def run():
            await manager._on_result(session, {'slot': slot, 'data': None}, asyncio.get_running_loop())
        asyncio.run(run())
        self.assertIn(slot, session.free_slots)

    def test_stop_and_frame_publication_are_ordered(self):
        from api.user_manager import UserManager, _Session
        manager = object.__new__(UserManager); manager.__init__()
        session = _Session('ordered', 'jump', proc=Mock(), in_q=queue.Queue(), out_q=queue.Queue())
        self.addCleanup(session.slots.destroy); manager._sessions['ordered'] = session
        entered, release = threading.Event(), threading.Event()
        original = attach_slot
        def slow_attach(*args):
            entered.set(); self.assertTrue(release.wait(2)); return original(*args)
        with patch('api.user_manager.attach_slot', side_effect=slow_attach):
            producer = threading.Thread(target=manager.push_frame, args=('ordered', np.zeros((2, 2, 3), np.uint8)))
            producer.start(); self.assertTrue(entered.wait(2))
            stopper = threading.Thread(target=manager._request_stop, args=(session,)); stopper.start()
            release.set(); producer.join(2); stopper.join(2)
        self.assertEqual([session.in_q.get_nowait()['type'] for _ in range(2)], ['task', 'stop'])
        self.assertFalse(manager.push_frame('ordered', np.zeros((2,2,3), np.uint8)))

    def test_old_consumer_disconnect_and_results_cannot_touch_new_session(self):
        from api.user_manager import UserManager, _Session
        manager = object.__new__(UserManager); manager.__init__()
        old_consumer, new_consumer = Mock(), Mock()
        old = _Session('replaced', 'jump', proc=Mock(), in_q=queue.Queue(), out_q=queue.Queue())
        self.addCleanup(old.slots.destroy)
        manager._consumers['replaced'] = new_consumer
        manager._sessions['replaced'] = object()
        with patch.object(manager, 'stop_sport_test') as stop:
            manager.unregister_consumer('replaced', old_consumer); stop.assert_not_called()
        async def run():
            await manager._on_result(old, {'data': {'state':'RESULT'}}, asyncio.get_running_loop())
        asyncio.run(run()); new_consumer.channel_layer.send.assert_not_called()

    def test_websocket_during_stop_keeps_finished_with_the_finalizer(self):
        from api.user_manager import UserManager, _Session

        async def run(sport_type):
            manager = object.__new__(UserManager); manager.__init__()
            process = Mock(); process.is_alive.return_value = True; process.pid = 12345
            session = _Session(f'closing_{sport_type}', sport_type, proc=process,
                               in_q=queue.Queue(), out_q=queue.Queue())
            manager._sessions[session.uid] = session
            entered, release = threading.Event(), threading.Event()
            original_await = manager._await_finished

            def await_finished(current):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test finalizer was not released')
                return original_await(current, timeout=.2)

            manager._await_finished = await_finished
            consumer = Mock(); consumer.uid = session.uid
            consumer.channel_layer.send = AsyncMock()
            stopping = asyncio.create_task(asyncio.to_thread(manager.stop_sport_test, session.uid))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                await manager.register_consumer(consumer)
                # Even a scheduled start callback must not create a second queue reader.
                manager._ensure_pump(session.uid)
                self.assertIsNone(session.pump)
                self.assertNotIn(session.uid, manager._consumers)
                consumer.channel_layer.send.assert_awaited_once_with(
                    consumer.channel_name, {'type': 'test_end'})
            finally:
                session.out_q.put({'type': 'finished'})
                release.set()
                await asyncio.wait_for(stopping, 3)
                if session.pump is not None:
                    session.pump.cancel()
                    await asyncio.gather(session.pump, return_exceptions=True)
                session.slots.destroy()

            self.assertTrue(session.finalized)
            process.terminate.assert_not_called()
            self.assertEqual(len(manager._idle[sport_type]), 1)
            self.assertIs(manager._acquire(sport_type)[0], process)

        for sport_type in ('jump', 'situp', 'pullup'):
            with self.subTest(sport=sport_type):
                asyncio.run(run(sport_type))

    def test_ten_immediate_restarts_wait_for_named_slot_cleanup(self):
        from api.user_manager import UserManager
        manager = object.__new__(UserManager); manager.__init__()
        async def run():
            manager._loop = asyncio.get_running_loop()
            pid = None
            async def until(predicate):
                deadline = time.monotonic() + 10
                while not predicate():
                    if time.monotonic() > deadline: self.fail('session lifecycle did not settle')
                    await asyncio.sleep(.01)
            try:
                for number in range(10):
                    self.assertTrue(await asyncio.to_thread(manager.start_sport_test, 'rapid', 'jump'))
                    session = manager._sessions['rapid']
                    if pid is None: pid = session.proc.pid
                    self.assertEqual(session.proc.pid, pid)
                    await until(lambda: session.ready)
                    self.assertTrue(manager.push_frame('rapid', np.zeros((4,5,3),np.uint8), 123+number))
                    await until(lambda: session.res_ok > 0)
                    await asyncio.to_thread(manager.stop_sport_test, 'rapid')
                    # Next iteration restarts immediately, without waiting for pump/finalize.
                await until(lambda: session.finalized)
                self.assertEqual(session.slots.names, [])
            finally:
                await asyncio.to_thread(manager.shutdown_all)
        with patch('api.user_manager.FACE_GATE_ENABLED', False), \
             patch('api.user_manager.run_user_worker', manager_fake_worker):
            asyncio.run(run())

    def test_task_timestamp_roundtrip_and_frame_not_serialized(self):
        import pickle
        task = SportTask('a', 0, 'jump', timestamp_ms=456)
        task.set_frame(np.zeros((3, 3, 3)))
        self.assertEqual(SportTask.from_dict(task.to_dict()).timestamp_ms, 456)
        self.assertNotIn('_frame', task.to_dict())
        with self.assertRaises(TypeError): pickle.dumps(task)


class LoadingContracts(unittest.TestCase):
    def test_missing_weights_do_not_invoke_ultralytics(self):
        with patch('ultralytics.YOLO') as load:
            with self.assertRaises(FileNotFoundError): model_runtime.create_yolo_segmenter('missing.pt', 'cpu')
            load.assert_not_called()

    def test_detection_weights_cannot_be_used_for_segmentation(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'model.pt'; checkpoint.write_bytes(b'test')
            with patch('ultralytics.YOLO', return_value=SimpleNamespace(task='detect', names={0: 'person'})):
                with self.assertRaises(ValueError): model_runtime.create_yolo_segmenter(checkpoint, 'cpu')
