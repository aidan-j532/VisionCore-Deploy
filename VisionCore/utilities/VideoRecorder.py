import cv2
import threading
import queue
import time
import os
import logging
from datetime import datetime

class VideoRecorder:
    def __init__(
        self,
        output_dir: str = "Video",
        fps: float = 30.0,
        codec: str = "mp4v",
        max_queue: int = 60,      # max buffered frames, drop if full to avoid memory bloat
        downsample: int = 1,      # write every Nth frame
    ):
        self.output_dir = output_dir
        self.fps = fps
        self.codec = codec
        self.max_queue = max_queue
        self.downsample = downsample

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._writer: cv2.VideoWriter | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._started = False
        self._frame_counter = 0
        self._dropped = 0
        self.logger = logging.getLogger(__name__)

        os.makedirs(output_dir, exist_ok=True)

    def start(self, width: int, height: int):
        if self._started:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))

        if not self._writer.isOpened():
            self.logger.error(f"VideoWriter failed to open: {filename}")
            self._writer = None
            return

        self._started = True
        self._stopped = False
        self._thread = threading.Thread(target=self._worker, daemon=True, name="VideoRecorder")
        self._thread.start()
        self.logger.info(f"Recording started: {filename} @ {width}x{height} {self.fps}fps")

    def write(self, frame):
        if not self._started or self._stopped or frame is None:
            return

        self._frame_counter += 1
        if self._frame_counter % self.downsample != 0:
            return  # skip this frame per downsample ratio

        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                self.logger.warning(
                    f"VideoRecorder queue full — dropped {self._dropped} frames total. "
                    "Consider increasing max_queue or downsample ratio."
                )

    def stop(self):
        if not self._started:
            return

        self._stopped = True
        self._queue.put(None)  # sentinel to unblock the worker

        if self._thread is not None:
            self._thread.join(timeout=10)

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        self.logger.info(
            f"Recording stopped. Total frames written: {self._frame_counter - self._dropped}, "
            f"dropped: {self._dropped}"
        )

    def _worker(self):
        while True:
            try:
                frame = self._queue.get(timeout=1.0)
            except queue.Empty:
                if self._stopped:
                    break
                continue

            if frame is None:  # sentinel
                break

            if self._writer is not None:
                self._writer.write(frame)