"""Bounded, cancellable CUDA admission shared by local text jobs on the same GPU."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile
import threading
import time


def memory_gib(file, field):
    for line in Path(file).read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError(f"Cannot determine {field} from {file}")


@contextmanager
def gpu_slot(model):
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one GPU with CUDA_VISIBLE_DEVICES")
    props = torch.cuda.get_device_properties(0)
    device = str(props.uuid)
    max_cap, min_cap, rss_cap = (26, 20, 72) if model == "glimmer" else (10, 9, 24)
    reserve = 5
    directory = Path(tempfile.gettempdir()) / f"particle-sliders-text-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    parent = os.getppid()
    with (directory / (device + ".lock")).open("a") as lock:
        deadline = time.monotonic() + 90
        while True:
            free = torch.cuda.mem_get_info(0)[0] / 2**30
            host = memory_gib("/proc/meminfo", "MemAvailable")
            if free >= min_cap + reserve and host >= 47:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pass
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"No shared GPU slot for {model} ({free:.1f} GiB GPU and {host:.1f} GiB host available)"
                )
            time.sleep(2)
        cap = min(max_cap, free - reserve)
        torch.cuda.set_per_process_memory_fraction(cap * 2**30 / props.total_memory, 0)
        stop = threading.Event()

        def guard():
            while not stop.wait(0.5):
                if (
                    torch.cuda.mem_get_info(0)[0] < reserve * 2**30
                    or os.getppid() != parent
                    or memory_gib("/proc/meminfo", "MemAvailable") < 47
                    or memory_gib("/proc/self/status", "VmRSS") > rss_cap
                ):
                    os.write(
                        2,
                        b"Local text worker stopped at a resource floor or parent exit.\n",
                    )
                    os._exit(75)

        thread = threading.Thread(target=guard, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
