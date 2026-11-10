"""Replay a recorded camera through the real spawn worker + shared memory.

Uses the production Jump adapter, without a face service (tested separately).
"""
import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from core.shm import SlotSet, attach_slot
from core.task import SportTask
from worker.run_user_worker import run_user_worker


def run(video, output):
    import cv2
    context = mp.get_context('spawn')
    iq, oq = context.Queue(), context.Queue()
    process = context.Process(target=run_user_worker, args=('jump', iq, oq))
    slots = SlotSet('replay_jump')
    free = list(slots.names)
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0: raise RuntimeError(f'Cannot open video: {video}')
    report = {'video': str(video), 'face': 'disabled for video replay', 'states': [],
              'received': 0, 'submitted': 0, 'dropped': 0, 'tail_submitted': 0, 'results': 0}
    process.start(); iq.put({'type': 'start'})
    output.mkdir(parents=True, exist_ok=True)
    last = None
    last_input = None
    def receive(message):
        nonlocal last
        if message['type'] == 'error': raise RuntimeError(message['message'])
        if message['type'] != 'result': return
        free.append(message['slot']); report['results'] += 1
        data = message.get('data')
        if not data: return
        last = data
        if not report['states'] or report['states'][-1] != data['state']:
            report['states'].append(data['state']); print(data['state'], data.get('message'), flush=True)
        if data['state'] in ('RESULT', 'FAILED'):
            report['result'] = data
            if last_input is not None:
                shm, image = attach_slot(message['slot'], last_input.shape)
                cv2.imwrite(str(output / 'processed.png'), image.copy()); shm.close()
    try:
        ready = oq.get(timeout=120)
        if ready != {'type': 'loaded', 'value': True}: raise RuntimeError(str(ready))
        started = time.monotonic(); epoch = time.time(); index = -1; frame_id = 0
        next_sample = 0.0
        while True:
            ok, frame = capture.read(); index += 1
            if not ok: break
            seconds = index / fps
            if seconds + 1e-6 < next_sample: continue
            next_sample += 1 / 15
            report['received'] += 1
            time.sleep(max(0, started + seconds - time.monotonic()))
            while True:
                try: receive(oq.get_nowait())
                except queue.Empty: break
            if last and last['state'] in ('RESULT', 'FAILED'): break
            h, w = frame.shape[:2]; scale = min(1, 1280 / w, 720 / h)
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
            last_input = frame.copy()
            if not free: report['dropped'] += 1; continue
            slot = free.pop(0); shm, shared = attach_slot(slot, frame.shape)
            shared[:] = frame; shm.close()
            task = SportTask('replay_jump', frame_id, 'jump', slot, frame.shape, {},
                             timestamp_ms=round((epoch + seconds) * 1000))
            iq.put({'type': 'task', 'task': task.to_dict()})
            frame_id += 1; report['submitted'] += 1
        # Keep delivering the last frame so asynchronous measurement can publish its result.
        frame = last_input
        deadline = time.monotonic() + 60
        while frame is not None and (last is None or last['state'] not in ('RESULT', 'FAILED')) and time.monotonic() < deadline:
            time.sleep(1 / 15)
            if free:
                slot = free.pop(0); shm, shared = attach_slot(slot, frame.shape)
                shared[:] = frame; shm.close()
                task = SportTask('replay_jump', frame_id, 'jump', slot, frame.shape, {},
                                 timestamp_ms=int(time.time() * 1000))
                iq.put({'type': 'task', 'task': task.to_dict()}); frame_id += 1; report['tail_submitted'] += 1
            try: receive(oq.get(timeout=.2))
            except queue.Empty: pass
        report['elapsed_seconds'] = time.monotonic() - started
        report['last'] = last
        iq.put({'type': 'stop'})
        while True:
            message = oq.get(timeout=10)
            if message['type'] == 'finished': break
            receive(message)
        iq.put({'type': 'stop'}); process.join(10)
        report['clean_exit'] = not process.is_alive() and process.exitcode == 0
    finally:
        capture.release()
        if process.is_alive(): process.terminate(); process.join(5)
        iq.close(); oq.close(); slots.destroy()
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.video, args.output)
    print(json.dumps({k: result.get(k) for k in ('states', 'received', 'submitted', 'dropped', 'clean_exit')}, ensure_ascii=False))
    sys.exit(0 if result.get('result', {}).get('state') == 'RESULT' and result.get('clean_exit') else 1)
