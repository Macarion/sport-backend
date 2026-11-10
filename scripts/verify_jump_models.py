"""Compare migrated YOLO masks with an explicit local reference, in separate processes."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def produce(args):
    import cv2
    import numpy as np
    if args.reference:
        sys.path.insert(0, str(args.reference / 'src'))
        from wcx_jump.segmentation import create_segmenter
        segmenter = create_segmenter('yolo11n-seg', args.reference / 'models/yolo11n-seg.pt', device='cuda')
        origin = sys.modules['wcx_jump.segmentation'].__file__
    else:
        sys.path.insert(0, str(ROOT / 'src'))
        from sports.jump.model_runtime import create_yolo_segmenter, YOLO_SEG_MODEL_PATH
        segmenter = create_yolo_segmenter(YOLO_SEG_MODEL_PATH, 'cuda')
        origin = sys.modules['sports.jump.model_runtime'].__file__
        assert not any(k == 'wcx_jump' or k.startswith('wcx_jump.') for k in sys.modules)
    report = json.loads(args.report.read_text(encoding='utf-8'))['result']['result']
    image = cv2.imread(report['outputs']['landing_frame'])
    result = segmenter.predict(image, report['segmentation']['prompt_box'])
    args.output.mkdir(parents=True, exist_ok=True)
    mask_path = args.output / 'mask.png'
    cv2.imwrite(str(mask_path), result['mask'])
    sys.path.insert(0, str(ROOT / 'src'))
    from sports.jump.model_runtime import get_algorithms
    measurement = get_algorithms().measure_jump_from_files(
        Path(report['outputs']['landing_frame']), mask_path, args.output,
        Path(report['outputs']['landing_frame']).parent / 'calib_frame.jpg')
    result.pop('mask')
    result['mask_sha256'] = hashlib.sha256(cv2.imread(str(mask_path), 0).tobytes()).hexdigest()
    result['score_cm'] = measurement.score_cm
    result['origin'] = origin
    assert not any(k == 'sam2' or k.startswith('sam2.') for k in sys.modules)
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        produce(args)
    else:
        if not args.reference: parser.error('--reference is required for the comparison')
        for mode in ('backend', 'reference'):
            command = [sys.executable, __file__, '--child', '--report', str(args.report), '--output', str(args.output / mode)]
            if mode == 'reference': command += ['--reference', str(args.reference)]
            subprocess.run(command, check=True)
        a, b = [json.loads((args.output / mode / 'result.json').read_text(encoding='utf-8')) for mode in ('backend','reference')]
        checks = {key: a[key] == b[key] for key in ('mask_sha256','best_mask_index','instance_selection','score_cm','iou_predictions')}
        result = {'checks': checks, 'passed': all(checks.values()), 'score_cm': a['score_cm'],
                  'backend_origin': a['origin'], 'reference_origin': b['origin'], 'mask_sha256': a['mask_sha256']}
        (args.output / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result)); sys.exit(0 if result['passed'] else 1)
