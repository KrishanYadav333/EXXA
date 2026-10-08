# Host-RAM leak probe (Kaggle script `krishanyadav333/ram-leak-probe`, versions 1 and 2), 2026-10-08

`tools/ram_leak_probe.py` trains the real U-Net on synthetic 64 px tiles (batch 4, 2 loader workers, no real data) and prints
main-process RSS, workers' RSS, shared memory and free RAM every 1,000 iterations, one factor changed per configuration, each in
its own process. T4 x2.

| config | what differs | main RSS growth per 1,000 iterations |
|---|---|---|
| D | loader + `.to(device)`, no model | +0 MB |
| F | model, single GPU, plain MSE | +0 MB |
| A | model, single GPU, the real hybrid (MSE + SSIM) loss | +0 MB |
| H | model, single GPU, SSIM term alone | +0 MB |
| **B** | model, **DataParallel**, hybrid loss | **+1,442 MB** |
| **G** | model, **DataParallel**, plain MSE | **+1,442 MB** |

(A, B, C, E crashed in version 1 on a missing `pytorch_msssim`; C and E were not repeated.) Only DataParallel changes the slope; the
loss, the loader, shared memory and the model do not. About 1.4 MB per iteration, independent of image size.

This reproduces nb14 v11 exactly: the patch arm has 700 images x 8 patches = 5,600 items = 1,400 iterations/epoch at batch 4, and
1,400 x 1.4 MB = 1.9 GB/epoch (observed 1.93). Note the notebook's own print said "patch: 44,800": it multiplied `len(train_ds_patch)`
by `N_PATCHES` a second time (that length already includes it); fixed.

Not established: *why* DataParallel leaks (per-forward thread creation is the obvious suspect). The fix does not need it.
