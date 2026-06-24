import subprocess
import threading
import time
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

        self.ffmpeg_process = None  # 存储FFmpeg进程对象

        self.frame_queue = queue.Queue(maxsize=30)  # 创建帧队列，最大容量30

        self.read_thread = None  # 存储读帧线程对象

        self.running = False  # 运行状态标志

        self.time = [0] * 10

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
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass

    def handle_video_stream(self, video_stream):
        if not self.ffmpeg_process:
            return

        try:
            if self.time[0] == 0:
                self.time[0] = time.time()
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
                "-f",
                "mp4",
                "-i",
                "pipe:0",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-probesize",
                "32",
                "-analyzeduration",
                "0",
                "-map",
                "0:v:0",
                "-threads",
                "auto",
                # "-vf", f"scale={self.width}:{self.height},fps={self.fps}",
                # "-vf", f"scale={self.width}:{self.height}",
                # "-r", str(self.fps),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                # "-pix_fmt", "yuv420p",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # subprocess.DEVNULL,
            bufsize=0,
        )

    def read_frames(self):

        frame_index = 0

        while self.running:

            raw_parts = []
            raw_size = 0

            while raw_size < self.frame_size:

                part = self.ffmpeg_process.stdout.read(self.frame_size - raw_size)

                if not part:
                    break

                if self.time[1] == 0:
                    self.time[1] = time.time()
                    print(self.time)

                raw_parts.append(part)

                raw_size += len(part)

            raw = b"".join(raw_parts)

            if len(raw) < self.frame_size:

                print(f"读帧结束，剩余数据大小: {len(raw)} bytes")

                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, self.channel)
            )

            frame_index += 1

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

        print("FFmpeg读帧线程退出")

        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass


class FFmpegVideoEncoder:

    def __init__(self, width, height, fps):

        self.width = width
        self.height = height
        self.fps = fps

        self.frame_size = width * height * 3

        self.running = False

        self.ffmpeg_process = None

        self.packet_queue = queue.Queue(maxsize=100)

        self.read_thread = None

    def start(self):

        self.ffmpeg_process = self.start_ffmpeg()

        self.running = True

        self.read_thread = threading.Thread(target=self.read_packets, daemon=True)

        self.read_thread.start()

    def stop(self):

        self.running = False

        if self.ffmpeg_process:

            try:
                if self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.close()
            except:
                pass

            try:
                self.ffmpeg_process.terminate()
            except:
                pass

            try:
                self.ffmpeg_process.wait(timeout=3)
            except:
                self.ffmpeg_process.kill()

        if self.read_thread:
            self.read_thread.join(timeout=3)

        try:
            self.packet_queue.put_nowait(None)
        except:
            pass

    def encode_frame(self, frame):

        if not self.ffmpeg_process:
            return

        try:

            self.ffmpeg_process.stdin.write(frame.tobytes())

        except BrokenPipeError:
            print("encoder ffmpeg exited")

        except Exception as e:
            print("encoder write error:", e)

    def read_packets(self):

        while self.running:

            try:

                chunk = self.ffmpeg_process.stdout.read(32768)

                if not chunk:
                    break

                if self.packet_queue.full():

                    try:
                        self.packet_queue.get_nowait()
                    except:
                        pass

                self.packet_queue.put_nowait(chunk)

            except Exception as e:

                print("encoder read error:", e)

                break

        print("FFmpeg encoder thread exit")

        try:
            self.packet_queue.put_nowait(None)
        except:
            pass

    def start_ffmpeg(self):

        ffmpeg_exe = shutil.which("ffmpeg")

        if ffmpeg_exe is None:

            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        return subprocess.Popen(
            [
                ffmpeg_exe,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libvpx",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                "-b:v",
                "1M",
                "-f",
                "webm",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
