"""Same process-HBM sampling as the Tensor benchmark, without importing torch_npu.

Keeping this helper independent lets Faiss use its own CANN runtime without
loading the Tensor environment's NPU extension or changing an active benchmark.
"""
import os
import re
import subprocess
import threading

PROCESS_ROW = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\d+)\s*\|$", re.M)


def process_hbm_mib(output, pid):
    matches = [(int(npu), int(chip), int(memory))
               for npu, chip, process, _name, memory in PROCESS_ROW.findall(output)
               if int(process) == pid]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one npu-smi process row for PID {pid}; found {matches}")
    return matches[0]


class NpuProcessMemorySampler:
    def __init__(self, interval=0.05):
        self.interval, self.pid = interval, os.getpid()
        self.peak_bytes = self.samples = 0
        self.physical_device = None
        self._stop = threading.Event()
        self._thread = self._error = None

    def sample(self):
        output = subprocess.run(["npu-smi", "info"], check=True, capture_output=True,
                                text=True, timeout=15).stdout
        npu, chip, memory = process_hbm_mib(output, self.pid)
        if self.physical_device is not None and self.physical_device != (npu, chip):
            raise RuntimeError("npu-smi process moved to a different physical NPU")
        self.physical_device = npu, chip
        self.peak_bytes = max(self.peak_bytes, memory * (1 << 20))
        self.samples += 1

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except BaseException as error:
                self._error = error
                self._stop.set()

    def __enter__(self):
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()
        if self._error is not None:
            raise RuntimeError("npu-smi background HBM sampling failed") from self._error
