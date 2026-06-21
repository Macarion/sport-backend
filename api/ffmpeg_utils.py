import subprocess
import threading
import numpy as np
import queue
import shutil

# 视频参数

class FFmpegVideoHandler:

    """
    FFmpegVideoHandler类用于处理视频流，通过FFmpeg进行视频解码和格式转换。
    它使用多线程方式读取FFmpeg处理后的视频帧，并通过队列管理帧数据。
    """
    def __init__(self, width, height, fps, channel=3):


        """
        初始化FFmpegVideoHandler实例
        """

        self.width = width
        self.height = height
        self.fps = fps
        self.channel = channel
        self.frame_size = width * height * channel

        self.ffmpeg_process = None # 存储FFmpeg进程对象

        self.frame_queue = queue.Queue(maxsize=30)  # 创建帧队列，最大容量30
        self.frame_queue_lock = threading.Lock()  # 帧队列的锁，用于线程同步

        self.read_thread = None  # 存储读帧线程对象

        self.running = False  # 运行状态标志

    def start(self):

        """
        启动FFmpeg处理和读帧线程
        """
        self.ffmpeg_process = self.start_ffmpeg()  # 启动FFmpeg进程
        print("ffmpeg started", self.ffmpeg_process)  # 打印FFmpeg进程信息

        self.running = True  # 设置运行状态为True

        # 创建并启动读帧线程，设置为守护线程
        self.read_thread = threading.Thread(target=self.read_frames, daemon=True)
        self.read_thread.start()

    def stop(self):

        """
        停止FFmpeg处理和读帧线程
        """
        self.running = False

        # 通知ffmpeg输入结束  # 设置运行状态为False
        if self.ffmpeg_process and self.ffmpeg_process.stdin:
            try:
                self.ffmpeg_process.stdin.close()
            except Exception:
                pass

        # 等待读帧线程退出
        if self.read_thread:
            self.read_thread.join(timeout=3)

        # 结束ffmpeg进程
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()

            try:
                self.ffmpeg_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()

        # 通知消费者退出
        try:
            self.frame_queue_lock.acquire()
            self.frame_queue.put_nowait(None)
            self.frame_queue_lock.release()
        except queue.Full:
            pass

    def handle_video_stream(self, video_stream):
        if not self.ffmpeg_process:
            return

        try:
            self.ffmpeg_process.stdin.write(video_stream)
            self.ffmpeg_process.stdin.flush()

        except BrokenPipeError:
            print("ffmpeg进程已退出")

        except Exception as e:
            print(f"写入ffmpeg失败: {e}")

    def start_ffmpeg(self):
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe is None:
            try:
                import imageio_ffmpeg

                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as exc:
                raise RuntimeError(
                    "未找到 ffmpeg，且 imageio-ffmpeg 不可用，无法解码前端视频流"
                ) from exc

        return subprocess.Popen(
            [
                ffmpeg_exe,

                "-f", "webm",
                "-i", "pipe:0",

                "-map", "0:v:0",

                "-vf", f"scale={self.width}:{self.height},fps={self.fps}",

                "-f", "rawvideo",
                "-pix_fmt", "bgr24",

                "pipe:1",
            ],

            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,

            stderr=subprocess.DEVNULL,

            bufsize=0
        )

    def read_frames(self):

        frame_index = 0

        while self.running:

            raw_parts = []
            raw_size = 0

            while raw_size < self.frame_size:

                part = self.ffmpeg_process.stdout.read(
                    self.frame_size - raw_size
                )

                if not part:
                    break

                raw_parts.append(part)

                raw_size += len(part)

            raw = b"".join(raw_parts)

            if len(raw) < self.frame_size:

                print(
                    f"读帧结束，剩余数据大小: {len(raw)} bytes"
                )

                break

            frame = np.frombuffer(
                raw,
                dtype=np.uint8
            ).reshape(
                (self.height, self.width, self.channel)
            )

            frame_index += 1

            self.frame_queue_lock.acquire()
            # 队列满则丢弃最旧帧
            if self.frame_queue.full():

                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)

            except queue.Full:
                pass
            self.frame_queue_lock.release()

        print("FFmpeg读帧线程退出")

        try:
            self.frame_queue_lock.acquire()
            self.frame_queue.put_nowait(None)
            self.frame_queue_lock.release()
        except queue.Full:
            pass
