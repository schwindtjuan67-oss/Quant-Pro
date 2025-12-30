import time
import threading

class HeartbeatMonitor:
    def __init__(self, timeout_seconds=20):
        self.last_tick = time.time()
        self.timeout = timeout_seconds
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.last_tick = time.time()

    def is_alive(self):
        with self.lock:
            return (time.time() - self.last_tick) < self.timeout
