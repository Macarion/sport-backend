import asyncio
import json
import os
import numpy as np
from aioice import Candidate
from av import VideoFrame
from channels.generic.websocket import AsyncWebsocketConsumer
from aiortc.rtcicetransport import candidate_from_aioice
from aiortc import MediaStreamError, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from multiprocessing.managers import BaseManager
from shared_memory_pool import SharedMemoryPool
from config import FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS, get_logger

log = get_logger('WebSocket')

MANAGER_HOST = os.environ.get('MANAGER_HOST', '127.0.0.1')
MANAGER_PORT = int(os.environ.get('MANAGER_PORT', 50000))
AUTH_KEY = bytes.fromhex(os.environ.get('AUTH_KEY', '626f6479746573745f7365637265745f6b6579'))
SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


class QueueManager(BaseManager):
    pass


QueueManager.register('get_task_queue')
QueueManager.register('get_command_queue')
QueueManager.register('get_result_queue')
QueueManager.register('get_shm_pool')


def get_manager():
    manager = QueueManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=AUTH_KEY)
    manager.connect()
    return manager


class ProcessVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.processed_frame = None

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        if self.processed_frame is None:
            frame = np.zeros(SHAPE, dtype=np.uint8)
        else:
            frame = self.processed_frame
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = pts
        vf.time_base = time_base
        return vf

    def push_frame(self, img_bgr):
        self.processed_frame = img_bgr


class WebRTCConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.uid = self.scope["url_route"]["kwargs"]["uid"]
        self.project = self.scope["url_route"]["kwargs"].get("project", "situp")
        self.group_name = f"webrtc_{self.uid}"

        if self.uid is None:
            await self.close()
            return

        await self.channel_layer.group_add(self.group_name, self.channel_name)

        self.pc = RTCPeerConnection()
        self.output_track = ProcessVideoTrack()
        self.pc.addTrack(self.output_track)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                asyncio.create_task(self.handle_video(track))

        self._result_task = None
        self._setup_ipc()

        await self.accept()

    def _setup_ipc(self):
        self.manager = get_manager()
        self.cmd_queue = self.manager.get_command_queue()
        self.task_queue = self.manager.get_task_queue(self.project)
        self.result_queue = self.manager.get_result_queue()
        self.shm_pool = self.manager.get_shm_pool()

        self.cmd_queue.put(('register', self.project, 1))
        log.info(f"用户注册: uid={self.uid}, project={self.project}")

        self._result_task = asyncio.create_task(self._poll_results())

    async def _poll_results(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                result = await loop.run_in_executor(
                    None, lambda: self.result_queue.get(timeout=0.05)
                )
                if result is None:
                    continue

                shm_name, user_id, eval_data = result
                if str(user_id) != str(self.uid):
                    self.shm_pool.release_block(shm_name)
                    continue

                frame = await loop.run_in_executor(
                    None, lambda: self._read_and_release_frame(shm_name)
                )
                if frame is not None:
                    self.output_track.push_frame(frame)

                await self.send(text_data=json.dumps({
                    "type": "eval_result",
                    "data": eval_data
                }))

            except Exception:
                await asyncio.sleep(0.01)

    def _read_and_release_frame(self, shm_name):
        shm, arr = SharedMemoryPool.attach(shm_name, SHAPE)
        frame = arr.copy() if arr is not None else None
        shm.close()
        self.shm_pool.release_block(shm_name)
        return frame

    async def disconnect(self, code):
        if self._result_task:
            self._result_task.cancel()
            try:
                await self._result_task
            except asyncio.CancelledError:
                pass

        if hasattr(self, 'cmd_queue') and self.cmd_queue:
            self.cmd_queue.put(('unregister', self.project, -1))
            log.info(f"用户注销: uid={self.uid}, project={self.project}")

        if hasattr(self, 'pc') and self.pc:
            await self.pc.close()

        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)

        if data["type"] == "offer":
            offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await self.pc.setRemoteDescription(offer)
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            await self.send(text_data=json.dumps({
                "type": "answer",
                "sdp": self.pc.localDescription.sdp
            }))

        elif data["type"] == "ice":
            cand = data.get("candidate")
            if not cand:
                return
            candidate = Candidate.from_sdp(cand.get("candidate"))
            icecandidate = candidate_from_aioice(candidate)
            icecandidate.sdpMid = cand.get("sdpMid")
            icecandidate.sdpMLineIndex = cand.get("sdpMLineIndex")
            await self.pc.addIceCandidate(icecandidate)

    async def handle_video(self, track):
        loop = asyncio.get_event_loop()
        frame_id = 0

        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break

            frame_nd = frame.to_ndarray(format="bgr24")
            frame_bytes = frame_nd.tobytes()

            success = await loop.run_in_executor(
                None, lambda: self._send_frame(frame_bytes, frame_id)
            )
            frame_id += 1

    def _send_frame(self, frame_bytes, frame_id):
        if self.task_queue is None or self.shm_pool is None:
            return False

        shm_name = self.shm_pool.acquire_block(timeout=0.05)
        if shm_name is None:
            log.warning(f"共享内存池已满，丢弃帧 uid={self.uid}")
            return False

        shm, arr = SharedMemoryPool.attach(shm_name, SHAPE)
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(SHAPE)
        arr[:] = frame
        shm.close()

        self.task_queue.put((shm_name, self.uid, frame_id))
        return True

    async def send_data(self, event):
        text_data = event["text"]
        await self.send(text_data=json.dumps({
            "type": "data",
            "data": text_data
        }))

    async def close_client(self, text_data=None, bytes_data=None):
        await self.disconnect(None)
