"""Jump adapter for the shared worker; models live for one process, sessions do not."""
import copy
import time

import numpy as np

from core.config import BACKEND_ROOT
from sports.base import BaseSport
from .model_runtime import get_algorithms, get_model_bundle, release_model_bundle
from .painting import draw_painting_on_image
from .session import FastReadyWebSession


class SportJump(BaseSport):
    def __init__(self):
        self.session = None
        self._face_client = None
        self._models = None
        self._result_frame = None
        self._shape = None

    def load(self):
        self._models = get_model_bundle()
        get_algorithms()

    def reset(self):
        self.end_session()
        self._result_frame = None
        self._shape = None

    def process(self, frame, config, frame_id, uid=0, timestamp_ms=None):
        if self._models is None:
            raise RuntimeError('SportJump.load() must complete first')
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Jump requires uint8 BGR frames')
        if frame.shape[0] > 720 or frame.shape[1] > 1280:
            raise ValueError('Jump video must be at most 1280x720')
        if self._shape is not None and frame.shape != self._shape:
            raise ValueError('Jump camera resolution changed; restart calibration')
        self._shape = frame.shape
        if self.session is None:
            self.session = FastReadyWebSession(str(uid), BACKEND_ROOT / 'outputs',
                                               BACKEND_ROOT / 'temp', face_client=self._face_client)
        if self.session.uid != str(uid):
            raise RuntimeError('Different user supplied without resetting the session')
        timestamp_ms = int(time.time() * 1000) if timestamp_ms is None else timestamp_ms
        response = self.session.process_stream_frame(frame, timestamp_ms, frame_id)
        if response is None:
            response = self.session.status_response()
        with self.session.lock:
            if response['state'] == 'RESULT' and self._result_frame is None:
                landing = self.session.landing_sample
                if landing is not None:
                    self._result_frame = draw_painting_on_image(
                        landing.packet.image, response.get('painting', []))
        painting = self._result_frame
        if painting is None:
            painting = draw_painting_on_image(frame, response.get('painting', []))
        if painting.shape != frame.shape:
            raise ValueError('Jump result and input frame dimensions differ')
        frame[:] = painting
        # Pixels stay in SHM. Do not send base64 images or duplicate drawing commands per frame.
        payload = copy.deepcopy(response)
        payload.pop('painting', None)
        if payload.get('result'):
            payload['result'].pop('painting', None)
            payload['result'].pop('landing_frame_preview', None)
        return payload

    def end_session(self):
        if self.session is not None:
            self.session.stop()
            self.session = None
        self._result_frame = None
        self._shape = None

    def close(self):
        self.end_session()
        self._models = None
        release_model_bundle()
