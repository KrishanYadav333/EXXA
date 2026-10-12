"""
Kaggle probe: which part of train_unet's loop leaks host RAM in the MAIN process?

nb14 v11 (PROGRESS.md 2026-10-08): `main` RSS grew 1.93 GB/epoch (workers flat) on the 64 px patch arm
(11,200 iterations of batch 4 per epoch), with glibc tuning and pinned memory already off. This runs the
same loop on synthetic tensors with one factor changed at a time and prints main RSS / free RAM / shared
memory every 1,000 iterations, so the slope per 1,000 iterations says which factor owns it.

Each config runs in its own subprocess (a leak in one cannot move the baseline of the next).
Run: python tools/ram_leak_probe.py            (all configs)
     python tools/ram_leak_probe.py --cfg A    (one)
"""
import os, sys, gc, time, argparse, subprocess

CFGS = {
    # name: (workers, data_parallel, use_model, sharing_strategy, loss)
    # First run (2026-10-08): D (loader only) and F (model + MSE, single GPU) were flat; A, B, C, E crashed on a
    # missing pytorch_msssim. Remaining suspects are the SSIM loss term and DataParallel, one at a time:
    "A": (2, False, True, None, "hybrid"),   # the real loss, single GPU
    "B": (2, True,  True, None, "hybrid"),   # the real loss, DataParallel (what train_unet does on T4x2)
    "G": (2, True,  True, None, "mse"),      # DataParallel, plain MSE
    "H": (2, False, True, None, "ssim"),     # SSIM term alone, single GPU
}
SIZE, BS, ITERS, EVERY = 64, 4, 6000, 1000


def meminfo():
    kb = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            kb[k] = int(v.split()[0])
    return kb


def rss_gb(pid):
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except OSError:
        return 0.0


def kids_gb():
    me, tot = os.getpid(), 0.0
    for e in os.listdir("/proc"):
        if e.isdigit():
            try:
                with open(f"/proc/{e}/stat") as f:
                    if int(f.read().rsplit(")", 1)[1].split()[1]) == me:
                        tot += rss_gb(e)
            except (OSError, ValueError, IndexError):
                pass
    return tot


def run_one(name):
    workers, dp, use_model, strategy, loss_name = CFGS[name]
    import torch
    from torch.utils.data import Dataset, DataLoader

    sys.path.insert(0, "/kaggle/working/EXXA/DENOISING_DIFFUSION")
    if strategy:
        torch.multiprocessing.set_sharing_strategy(strategy)
    dev = torch.device("cuda")
    ngpu = torch.cuda.device_count()

    class Syn(Dataset):
        def __len__(self):
            return 44800

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return torch.rand(1, SIZE, SIZE, generator=g), torch.rand(1, SIZE, SIZE, generator=g)

    loader = DataLoader(Syn(), batch_size=BS, shuffle=True, num_workers=workers, pin_memory=False,
                        persistent_workers=workers > 0)
    model = opt = crit = None
    if use_model:
        from src.training.architectures import build_model, forward_fn
        fwd = forward_fn("unet")
        net = build_model("unet", base_channels=48, channel_multipliers=(1, 2, 4, 8), use_beam=False,
                          n_neighbors=0, out_channels=1, latent_dim=128).to(dev)
        model = torch.nn.DataParallel(net) if (dp and ngpu > 1) else net
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        if loss_name == "hybrid":
            from src.training.sweep import LOSS_REGISTRY
            crit = LOSS_REGISTRY["hybrid"](alpha=0.8877, beta=0.1123)
        elif loss_name == "ssim":
            from pytorch_msssim import ssim as _ssim
            crit = lambda p, c: (1.0 - _ssim(p, c, data_range=1.0, size_average=True),)
        else:
            crit = lambda p, c: (torch.nn.functional.mse_loss(p, c),)

    print(f"[{name}] workers={workers} dp={dp and ngpu > 1} (gpus={ngpu}) model={use_model} "
          f"strategy={strategy} loss={loss_name}", flush=True)
    rows, it, t0 = [], 0, time.time()
    while it < ITERS:
        for d, c in loader:
            d, c = d.to(dev), c.to(dev)
            if use_model:
                pred, _ = fwd(model, d, None)
                total = crit(pred, c)[0]
                opt.zero_grad()
                total.backward()
                opt.step()
                total.item()
            it += 1
            if it % EVERY == 0:
                m = meminfo()
                rows.append((it, m["MemAvailable"] / 1048576, rss_gb(os.getpid()), kids_gb(), m["Shmem"] / 1048576))
                print(f"[{name}] it {it:6d} | free {rows[-1][1]:5.2f} | main {rows[-1][2]:5.2f} | "
                      f"workers {rows[-1][3]:5.2f} | shmem {rows[-1][4]:5.2f} | {time.time() - t0:5.0f}s", flush=True)
            if it >= ITERS:
                break
    a, b = rows[1], rows[-1]       # skip the first reading (warm-up allocations)
    per = (b[2] - a[2]) / (b[0] - a[0]) * 1000
    perfree = (a[1] - b[1]) / (b[0] - a[0]) * 1000
    perwk = (b[3] - a[3]) / (b[0] - a[0]) * 1000
    pershm = (b[4] - a[4]) / (b[0] - a[0]) * 1000
    print(f"[{name}] RESULT per 1000 iters: main {per * 1000:+.0f} MB | workers {perwk * 1000:+.0f} MB | "
          f"shmem {pershm * 1000:+.0f} MB | free {-perfree * 1000:+.0f} MB   "
          f"(nb14: 11,200 iters/epoch at 1.93 GB = ~170 MB per 1000)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg")
    a = ap.parse_args()
    if a.cfg:
        run_one(a.cfg)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch_msssim"], check=False)
        if not os.path.exists("/kaggle/working/EXXA"):
            subprocess.run(["git", "clone", "--depth", "1", "--branch", "native600-loss-sweep",
                            "https://github.com/KrishanYadav333/EXXA.git", "/kaggle/working/EXXA"], check=True)
        for n in CFGS:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--cfg", n])
        print("PROBE DONE", flush=True)
