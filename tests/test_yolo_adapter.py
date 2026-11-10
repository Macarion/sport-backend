"""Realtime YOLO segmentation contracts; no weights, GPU or camera needed."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sports.jump.yolo_segmentation import YoloImageSegmenter, YoloSegOptions


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def prediction(boxes, masks, scores=None, classes=None):
    count = len(boxes)
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=Tensor(boxes), conf=Tensor(scores if scores is not None else [.9] * count),
                              cls=Tensor(classes if classes is not None else [0] * count)),
        masks=SimpleNamespace(data=Tensor(masks)),
    )


class YoloMaskContracts(unittest.TestCase):
    def setUp(self):
        self.image = np.full((10, 12, 3), [3, 5, 7], np.uint8)
        self.box = [2, 3, 6, 8]
        self.masks = np.zeros((2, 10, 12), np.uint8)
        self.masks[0, 3:8, 2:6] = 1
        self.masks[1, :2, 8:] = 1
        self.model = Mock()
        self.model.predict.return_value = [prediction([self.box, [8, 0, 12, 2]], self.masks, [.8, .99])]
        self.torch = SimpleNamespace(cuda=Mock())

    def adapter(self, **options):
        return YoloImageSegmenter(self.model, self.torch, {'device': 'cpu', 'backend': 'yolo11n-seg'},
                                  YoloSegOptions(**options))

    def test_full_bgr_original_coordinates_and_locked_person_not_bystander(self):
        result = self.adapter().predict(self.image, self.box, measure=True)
        np.testing.assert_array_equal(self.model.predict.call_args.args[0], self.image)
        np.testing.assert_array_equal(result['mask'], self.masks[0])
        self.assertEqual(result['mask'].dtype, np.uint8)
        self.assertEqual(result['best_mask_index'], 0)
        self.assertEqual(result['prompt_box'], self.box)
        self.assertEqual(result['instance_selection']['target_box_iou'], 1)
        self.assertEqual(result['instance_selection']['instance_box'], self.box)
        self.assertEqual(result['instance_selection']['detection_confidence'], .8)
        self.assertEqual(result['iou_predictions'], [])
        self.assertIn('not mask quality', result['quality_semantics'])
        self.assertGreaterEqual(result['timing_ms']['total'], 0)

    def test_inference_parameters_and_no_saving(self):
        adapter = self.adapter(imgsz=960, conf=.35, iou=.6, max_det=7, match_iou=.3)
        result = adapter.predict(self.image, self.box)
        self.assertEqual(self.model.predict.call_args.kwargs,
                         dict(imgsz=960, conf=.35, iou=.6, max_det=7, classes=[0],
                              device='cpu', half=False, retina_masks=True, verbose=False, save=False))
        self.assertEqual(result['model']['inference']['match_iou'], .3)
        self.assertNotIn('timing_ms', result)

    def test_roi_restores_mask_and_instance_box_to_full_image(self):
        self.model.predict.return_value = [prediction([[0, 0, 4, 5]], np.ones((1, 5, 4)))]
        result = self.adapter(input_mode='roi').predict(self.image, self.box)
        np.testing.assert_array_equal(self.model.predict.call_args.args[0], self.image[3:8, 2:6])
        np.testing.assert_array_equal(result['mask'], self.masks[0])
        self.assertEqual(result['instance_selection']['instance_box'], self.box)
        self.assertEqual(result['instance_selection']['crop_bounds'], self.box)

    def test_roi_rounds_outwards_and_clips(self):
        self.model.predict.return_value = [prediction([[0, 0, 6, 8]], np.ones((1, 8, 6)))]
        result = self.adapter(input_mode='roi').predict(self.image, [-1.5, -2, 5.1, 7.2])
        self.assertEqual(result['instance_selection']['crop_bounds'], [0, 0, 6, 8])
        self.assertEqual(int(result['mask'].sum()), 48)
        self.assertEqual(result['mask'].shape, (10, 12))

    def test_ties_use_confidence_then_stable_index(self):
        masks = np.ones((3, 10, 12))
        self.model.predict.return_value = [prediction([self.box] * 3, masks, [.5, .9, .9])]
        self.assertEqual(self.adapter().predict(self.image, self.box)['best_mask_index'], 2)

    def test_same_loaded_model_reused_across_attempts(self):
        adapter = self.adapter()
        adapter.predict(self.image, self.box)
        adapter.predict(self.image, self.box)
        self.assertEqual(self.model.predict.call_count, 2)
        self.assertIs(adapter.predictor, self.model)

    def test_no_target_no_class_match_or_low_confidence_fails(self):
        for boxes, scores, classes in (([[8, 0, 12, 2]], [.99], [0]),
                                       ([self.box], [.99], [1]), ([self.box], [.1], [0])):
            self.model.predict.return_value = [prediction(boxes, np.ones((1, 10, 12)), scores, classes)]
            with self.subTest(boxes=boxes, classes=classes, scores=scores), self.assertRaisesRegex(ValueError, '无匹配'):
                self.adapter().predict(self.image, self.box)

    def test_no_masks_is_explicit_failure(self):
        for result in ([], [SimpleNamespace(masks=None, boxes=None)]):
            self.model.predict.return_value = result
            with self.assertRaisesRegex(ValueError, '未检测到人体'):
                self.adapter().predict(self.image, self.box)

    def test_wrong_resolution_empty_and_nonfinite_mask_rejected(self):
        for masks in (np.ones((1, 5, 6)), np.zeros((1, 10, 12)), np.full((1, 10, 12), np.nan)):
            self.model.predict.return_value = [prediction([self.box], masks)]
            with self.subTest(shape=masks.shape), self.assertRaises(ValueError):
                self.adapter().predict(self.image, self.box)

    def test_invalid_input_rejected_before_prediction(self):
        for image, box in ((self.image.astype(float), self.box), (self.image[:, :, 0], self.box),
                           (self.image[:0], self.box), (self.image, [0, 1, 0, 5]),
                           (self.image, [1, 2, np.nan, 4]), (self.image, [20, 20, 30, 30]),
                           (self.image, [1, 2])):
            with self.assertRaises(ValueError):
                self.adapter().predict(image, box)
        self.model.predict.assert_not_called()

    def test_cuda_timing_and_fp16_forwarded(self):
        adapter = YoloImageSegmenter(self.model, self.torch, {'device': 'cuda:0'}, YoloSegOptions(half=True))
        adapter.predict(self.image, self.box, measure=True)
        self.assertEqual(self.torch.cuda.synchronize.call_count, 2)
        self.assertTrue(self.model.predict.call_args.kwargs['half'])



class YoloOptionsAndLoading(unittest.TestCase):
    def setUp(self):
        from sports.jump.model_runtime import create_yolo_segmenter
        self.create = create_yolo_segmenter
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        self.checkpoint = Path(folder.name) / 'weight.pt'
        self.checkpoint.write_bytes(b'fake weight for mocked loading')

    def test_invalid_programmatic_options(self):
        for options in ({'imgsz': 0}, {'imgsz': 640.0}, {'imgsz': True}, {'conf': float('inf')},
                        {'iou': -1}, {'match_iou': 0}, {'input_mode': 'x'}, {'half': 1}, {'max_det': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                YoloSegOptions(**options)

    def test_half_on_cpu_rejected_before_model_loading(self):
        with patch('ultralytics.YOLO') as load, self.assertRaisesRegex(ValueError, 'FP16'):
            self.create(self.checkpoint, 'cpu', YoloSegOptions(half=True))
        load.assert_not_called()

    def test_non_person_weights_rejected(self):
        with patch('ultralytics.YOLO', return_value=SimpleNamespace(task='segment', names={0:'cat'})):
            with self.assertRaises(ValueError): self.create(self.checkpoint, 'cpu')

    def test_factory_retains_model_options_and_metadata(self):
        model = Mock(task='segment', names={0:'person'})
        model.model.parameters.return_value = [SimpleNamespace(numel=lambda:123)]
        options = YoloSegOptions(imgsz=960)
        with patch('ultralytics.YOLO', return_value=model):
            result = self.create(self.checkpoint, 'cpu', options)
        self.assertIsInstance(result, YoloImageSegmenter)
        self.assertIs(result.predictor, model)
        self.assertIs(result.options, options)
        self.assertEqual(result.metadata['parameter_count'], 123)
        self.assertEqual(result.metadata['precision'], 'FP32')
        self.assertEqual(len(result.metadata['checkpoint_sha256']), 64)
