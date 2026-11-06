"""Extra LP labels for Ex4 in small saved chunks, with a memory and load failsafe.

Each chunk runs in its own capped process and is saved when finished, so a restart resumes at the next missing chunk.
A chunk starts only when there is room; if available memory falls below the floor during a chunk, that chunk is killed and retried later.
When all chunks exist they are merged with <model>_train.npz into <model>_train_extended.npz.
Usage (from scripts/): python label_chunks_ex4.py [n_chunks] [chunk_size] [n_cpu]
"""
import os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np

N, SIZE, CPU = (int(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((1, 14), (2, 50_000), (3, 4)))
START_GB, KILL_GB, MAX_LOAD = 8.0, 4.0, 12.0
OUT = Path("artifacts/relu_nov2026"); CH = OUT / "ex4_chunks"; CH.mkdir(parents=True, exist_ok=True)


def avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 2**20


def room():
    return avail_gb() >= START_GB and os.getloadavg()[0] <= MAX_LOAD


for i in range(N):
    f = CH / f"chunk_{i:02d}.npz"
    while not f.exists():
        while not room():
            print(f"{time.strftime('%T')} waiting: {avail_gb():.1f} GB available, load {os.getloadavg()[0]:.1f}", flush=True)
            time.sleep(60)
        tmp = CH / f"tmp_{i:02d}.npz"
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=6G", "--", sys.executable, "train_surrogates_relu.py", "--models", "model4_50_dof",
               "--extra-labels", str(SIZE), "--data-seed", str(1000 + i), "--chunk-out", str(tmp), "--n-cpu", str(CPU)]
        t = time.time()
        p = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        killed = False
        while p.poll() is None:
            time.sleep(10)
            if avail_gb() < KILL_GB:
                print(f"{time.strftime('%T')} chunk {i}: only {avail_gb():.1f} GB available, killing it", flush=True)
                os.killpg(p.pid, signal.SIGKILL); killed = True; p.wait(); break
        if not killed and p.returncode == 0 and tmp.exists():
            tmp.rename(f)
            print(f"{time.strftime('%T')} chunk {i} done in {time.time()-t:.0f}s", flush=True)
        else:
            print(f"{time.strftime('%T')} chunk {i} not saved (return code {p.returncode})", flush=True)
            time.sleep(60)
d = [np.load(OUT / "model4_50_dof_train.npz")] + [np.load(CH / f"chunk_{i:02d}.npz") for i in range(N)]
np.savez(OUT / "model4_50_dof_train_extended.npz", x=np.concatenate([e["x"] for e in d]), y=np.concatenate([e["y"] for e in d]))
print(f"{time.strftime('%T')} merged: {sum(len(e['y']) for e in d)} points", flush=True)
