"""Takeoff policy and surrounding session transitions, without model inference."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sports.jump.model_runtime import get_algorithms
from sports.jump.session import FastReadyConfig, FastReadyWebSession


class TakeoffContracts(unittest.TestCase):
    def setUp(self):
        self.core = get_algorithms().realtime_core
        self.people = []
        realtime = SimpleNamespace(
            detect_people=lambda *args: self.people,
            match_locked_person=self.core.match_locked_person,
            select_ready_person=self.core.select_ready_person,
            bbox_iou=self.core.bbox_iou,
            detect_takeoff=self.core.detect_takeoff,
            FramePacket=self.core.FramePacket,
            JumpFrameSample=self.core.JumpFrameSample,
        )
        for name, value in [('get_algorithms', SimpleNamespace(realtime_core=realtime)),
                            ('get_model_bundle', SimpleNamespace(yolo_model=None))]:
            p = patch('sports.jump.session.' + name, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        self.frame = np.zeros((720, 1280, 3), np.uint8)
        self.frame_id = 0

    def session(self, face=None, **options):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        session = FastReadyWebSession('athlete', Path(folder.name),
                                      config=FastReadyConfig(**options), face_client=face)
        self.addCleanup(session.stop)
        session.state = 'WAITING_READY'
        return session

    def feed(self, session, y2, timestamp_ms):
        self.people = [] if y2 is None else [self.core.PersonDetection(
            bbox=[100, 100, 300, y2], confidence=.9,
            bottom_center_image=np.array([200, y2]),
            bottom_center_world=np.array([0, 0]))]
        result = session.process_frame(self.frame, self.frame_id, timestamp_ms)
        self.frame_id += 1
        return result

    def arm(self, session, y2=600):
        for timestamp in (1000, 1100, 1200):
            self.feed(session, y2, timestamp)
        self.assertEqual(session.state, 'ARMED')

    def test_dynamic_ground_correction_then_normal_two_frame_takeoff(self):
        session = self.session()
        self.arm(session, 500)
        self.feed(session, 530, 1300)
        self.assertEqual(session.baseline_y2, 530)
        self.assertEqual(session.state, 'ARMED')
        self.feed(session, 490, 1400)
        self.assertEqual(session.state, 'ARMED')
        self.feed(session, 490, 1500)
        self.assertEqual(session.state, 'IN_JUMP')
        self.assertEqual(session.debug_info['takeoff_reason'], 'two_frame')

    def test_airborne_frames_do_not_lower_or_enter_baseline_window(self):
        session = self.session(single_frame_takeoff_ratio=0)
        self.arm(session)
        self.feed(session, 560, 1300)
        self.assertEqual(session.baseline_y2, 600)
        self.assertEqual(list(session.armed_baseline_y2_values), [600, 600, 600])

    def test_baseline_window_is_bounded_and_baseline_never_decreases(self):
        session = self.session()
        self.arm(session)
        for index, y in enumerate([610] + [600] * 7):
            self.feed(session, y, 1300 + index * 100)
        self.assertEqual(len(session.armed_baseline_y2_values), 5)
        self.assertEqual(session.baseline_y2, 610)
        self.assertEqual(session.state, 'ARMED')

    def test_strong_single_frame_boundary(self):
        for rise, state in [(74, 'ARMED'), (75, 'IN_JUMP')]:
            with self.subTest(rise=rise):
                session = self.session()
                self.arm(session)
                self.feed(session, 600 - rise, 1300)
                self.assertEqual(session.state, state)
                if state == 'IN_JUMP':
                    self.assertEqual(session.debug_info['takeoff_reason'], 'single_frame_strong')

    def test_existing_two_frame_threshold_is_preserved(self):
        session = self.session()
        self.arm(session)
        self.feed(session, 570, 1300)
        self.assertEqual(session.state, 'ARMED')
        self.feed(session, 570, 1400)
        self.assertEqual(session.state, 'IN_JUMP')

    def test_disabling_strong_trigger_keeps_two_frame_rule(self):
        session = self.session(single_frame_takeoff_ratio=0)
        self.arm(session)
        self.feed(session, 520, 1300)
        self.assertEqual(session.state, 'ARMED')
        self.feed(session, 520, 1400)
        self.assertEqual(session.state, 'IN_JUMP')

    def test_disabling_both_restores_fixed_baseline_behavior(self):
        session = self.session(dynamic_armed_baseline=False, single_frame_takeoff_ratio=0)
        self.arm(session, 500)
        for index, y in enumerate([530, 490, 490]):
            self.feed(session, y, 1300 + index * 100)
        self.assertEqual(session.baseline_y2, 500)
        self.assertEqual(session.state, 'ARMED')

    def test_lost_frame_does_not_bridge_two_frame_takeoff(self):
        session = self.session()
        self.arm(session)
        for y, ts in [(560, 1300), (None, 1400), (560, 1500)]:
            self.feed(session, y, ts)
        self.assertEqual(session.state, 'ARMED')
        self.feed(session, 560, 1600)
        self.assertEqual(session.state, 'IN_JUMP')

    def test_face_wait_seeds_baseline_only_after_admission_and_restarts_clock(self):
        face = Mock()
        face.face_recognition.side_effect = [False, True]
        face.status.return_value = {'state': 'checking'}
        session = self.session(face)
        for ts in (1000, 1100, 1200):
            self.feed(session, 600, ts)
        self.assertEqual(session.state, 'WAITING_READY')
        self.assertEqual(list(session.armed_baseline_y2_values), [])
        self.feed(session, 600, 15000)
        self.assertEqual(session.state, 'ARMED')
        self.assertEqual(session.state_started_at, 15)
        self.assertEqual(list(session.armed_baseline_y2_values), [600] * 3)
        face.face_recognition.side_effect = AssertionError('Face gate called during takeoff')
        self.feed(session, 525, 15100)
        self.assertEqual(session.state, 'IN_JUMP')

    def test_takeoff_timestamp_and_two_second_review_window(self):
        session = self.session()
        self.arm(session)
        self.feed(session, 525, 1300)
        self.assertEqual(session.takeoff_timestamp_sec, 1.3)
        self.assertEqual(session.state_started_at, 1.3)
        self.assertEqual(session.jump_samples[0].packet.timestamp, 1.3)
        with patch.object(session, '_start_processing') as start:
            self.feed(session, 600, 3299)
            start.assert_not_called()
            self.feed(session, 600, 3301)
            start.assert_called_once_with(3.301)

    def test_timeout_precedes_strong_trigger(self):
        session = self.session()
        self.arm(session)
        self.feed(session, 525, 11201)
        self.assertEqual(session.state, 'FAILED')
        self.assertIsNone(session.takeoff_timestamp_sec)

    def test_new_session_has_no_previous_baseline_samples(self):
        first = self.session()
        self.arm(first)
        self.feed(first, 640, 1300)
        first.stop()
        second = self.session()
        self.assertEqual(list(second.armed_baseline_y2_values), [])
        self.arm(second, 500)
        self.assertEqual(second.baseline_y2, 500)
        self.assertEqual(list(second.armed_baseline_y2_values), [500] * 3)


if __name__ == '__main__':
    unittest.main()
