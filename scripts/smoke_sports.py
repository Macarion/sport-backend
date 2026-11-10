"""Real model lifecycle/drawing smoke test; no face SDK or physical camera needed."""
import argparse
import json
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from core.registry import load_sport

def main(args):
    capture = cv2.VideoCapture(str(args.video)); ok, frame = capture.read(); capture.release()
    if not ok: raise ValueError('Cannot read video')
    frame = cv2.resize(frame, (1280,720))
    report = {}
    for name in ('jump','situp','pullup'):
        sport = load_sport(name)(); sport.set_face_client(None); sport.load()
        model = sport._models if name == 'jump' else sport._inferencer
        sessions = []; baseline = {t.ident for t in threading.enumerate()}
        for number in range(10):
            sport.reset(); image = frame.copy()
            data = sport.process(image, {}, 0, uid=f'{name}-{number}', timestamp_ms=int(time.time()*1000))
            if name == 'jump':
                assert data['uid'] == f'{name}-{number}'
                assert data['session_id'] not in sessions
                sessions.append(data['session_id'])
                session_threads = (sport.session._raw_encoder, sport.session._diag_writer)
            else: session_threads = ()
            assert (sport._models if name == 'jump' else sport._inferencer) is model
            sport.end_session()
            assert not any(t.is_alive() for t in session_threads)
            if number == 0:
                args.output.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.output / f'{name}.jpg'), image)
        sport.close()
        report[name] = {'sessions':10, 'same_model':True, 'session_threads_joined':True,
                        'remaining_new_threads':[t.name for t in threading.enumerate() if t.ident not in baseline],
                        'drawing_changed_pixels':int(np.count_nonzero(image != frame))}
    assert not any(k == 'wcx_jump' or k.startswith('wcx_jump.') or k == 'sam2' or k.startswith('sam2.') for k in sys.modules)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--video',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
