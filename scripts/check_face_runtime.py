"""Validate deployed face models and IPC without changing the supplied registry."""
import argparse
import hashlib
import json
import multiprocessing as mp
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from core.config import FACE_DB_PATH, FACE_MODEL_DIR
from core.face_client import FaceClient
from core.face_service import run_face_service


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(args):
    from face_sdk import init_sdk, register, recognize, is_registered, close
    import onnxruntime as ort

    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read the supplied test image')
    before = digest(FACE_DB_PATH)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'onnxruntime': ort.__version__, 'providers': ort.get_available_providers(),
              'image': str(args.image), 'registry': str(FACE_DB_PATH)}
    init_sdk(model_dir=FACE_MODEL_DIR, db_path=FACE_DB_PATH)
    try:
        assert is_registered(args.student_id), 'Expected supplied student is not registered'
        matched, similarity, box = recognize(args.student_id, image)
        assert box is not None and np.isfinite(similarity), 'Face detection / embedding failed'
        report['supplied_sample'] = {'matched': bool(matched), 'similarity': similarity,
                                     'face_detected': True}
    finally:
        close()

    # Exercise registration only in a disposable database, never the real registry.
    with tempfile.TemporaryDirectory() as folder:
        init_sdk(model_dir=FACE_MODEL_DIR, db_path=Path(folder) / 'faces.db')
        try:
            assert register('fixture', [image])
            assert is_registered('fixture')
            matched, similarity, box = recognize('fixture', image)
            assert matched and similarity > .99
            assert recognize('fixture', np.zeros_like(image))[0] is False
            report['temporary_registration'] = {'passed': True, 'self_similarity': similarity,
                                                 'blank_image_rejected': True}
        finally:
            close()

    context = mp.get_context('spawn')
    requests = context.Queue()
    proc = context.Process(target=run_face_service, args=(requests,))
    client = FaceClient(requests, timeout=60)
    proc.start()
    try:
        report['service_ipc'] = []
        cases = [('missing-fixture-uid', image, False),
                 (args.student_id, np.zeros_like(image), True),
                 (args.student_id, image, True)]
        for uid, frame, registered in cases:
            client.reset()
            assert client.send(uid, frame)
            deadline = time.monotonic() + 60
            reply = None
            while reply is None and time.monotonic() < deadline:
                assert proc.is_alive(), 'Face service exited unexpectedly'
                reply = client.poll()
                if reply is None:
                    time.sleep(.05)
            assert reply is not None and not reply.get('timeout'), 'Face service timed out'
            assert reply['registered'] is registered
            if not registered:
                assert reply['ok'] is True
            elif not np.any(frame):
                assert reply['ok'] is False
            report['service_ipc'].append({k: reply[k] for k in ('registered', 'ok', 'conf')})
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(5)
        client._recv_end.close()
        client._send_end.close()
        requests.close()
    assert digest(FACE_DB_PATH) == before, 'The supplied registry was modified'
    with sqlite3.connect(f'{FACE_DB_PATH.as_uri()}?mode=ro', uri=True) as connection:
        report['registry_rows'] = connection.execute('SELECT COUNT(*) FROM faces').fetchone()[0]
    report.update(passed=True, registry_unchanged=True)
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--student-id', default='23281166')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/validation/face-setup')
    main(parser.parse_args())
