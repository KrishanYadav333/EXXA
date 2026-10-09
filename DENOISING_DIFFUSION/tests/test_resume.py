"""
train_unet must (1) stop cleanly before a session deadline, leaving a resumable state and NO checkpoint / score,
(2) continue from the epoch after the saved one in a later call, (3) refuse a state written by a different configuration.

Run: PYTHONPATH=. python3 tests/test_resume.py
"""
import os, sys, tempfile, time
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.training.sweep as sw


class Toy(Dataset):
    def __init__(self, n=8, size=32):
        g = torch.Generator().manual_seed(0)
        self.c = torch.rand(n, 1, size, size, generator=g)
        self.d = (self.c + 0.2 * torch.randn(n, 1, size, size, generator=g)).clamp(0, 1)

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        return self.d[i], self.c[i]


tmp = tempfile.mkdtemp()
rp, ck = os.path.join(tmp, "arm.resume"), os.path.join(tmp, "arm.pth")
kw = dict(base_channels=8, channel_multipliers=(1, 2), min_epochs=1, max_epochs=6, patience=100, batch_size=4,
          verbose=False, resume_path=rp, ckpt_path=ck, min_free_ram_gb=0)
sw._free_ram_gb = lambda: None
saved = []

# 1. a deadline that only fits ~one epoch: interrupted after it, state on disk, no checkpoint
t0 = time.time()
res = sw.train_unet(Toy(), Toy(), "cpu", deadline=time.time() + 0.01, epoch_callback=saved.append, **kw)
assert res.get("interrupted") is True and res["epochs_run"] == 1, res
assert os.path.exists(rp) and not os.path.exists(ck), "an interrupted arm must leave a state and no checkpoint"
assert saved == [rp], saved
print(f"interrupt   : stopped after epoch {res['epochs_run']}, state saved, no checkpoint, callback fired")

# 2. resume: continues at epoch 2 and finishes all 6 epochs
res = sw.train_unet(Toy(), Toy(), "cpu", **kw)
assert not res.get("interrupted") and res["epochs_run"] == 6, res.get("epochs_run")
assert os.path.exists(ck) and res["psnr"] > 0
print(f"resume      : ran epochs 2-6 to the end ({res['epochs_run']} total), checkpoint written, PSNR {res['psnr']:.2f}")

# 3. a state from another configuration is not adopted
rp2 = os.path.join(tmp, "other.resume")
sw.train_unet(Toy(), Toy(), "cpu", **{**kw, "resume_path": rp2, "ckpt_path": None, "max_epochs": 2})
res = sw.train_unet(Toy(), Toy(), "cpu", **{**kw, "resume_path": rp2, "ckpt_path": None, "lr": 5e-4})
assert res["epochs_run"] == 6, "different lr: must start fresh, not continue the old run"
print("fingerprint : a state written with a different lr was discarded, run started fresh")

# 4. a truncated state file (killed mid-write by an earlier session) must not crash the next session
rp3 = os.path.join(tmp, "broken.resume")
open(rp3, "wb").write(b"not a torch file")
res = sw.train_unet(Toy(), Toy(), "cpu", **{**kw, "resume_path": rp3, "ckpt_path": None, "max_epochs": 2})
assert res["epochs_run"] == 2
print("corrupt     : unreadable state ignored, run started fresh")
print("PASS")
