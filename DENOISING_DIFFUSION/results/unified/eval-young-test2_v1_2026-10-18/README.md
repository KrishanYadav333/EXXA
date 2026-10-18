# eval-young-test2_v1_2026-10-18

Recipe B on the 8 more held-out Young et al. simulations (TEST2), 300 and 600 px, PV included. 16 cases, none failed, 52.8 min.
**It ran under the wrong kernel name:** `chiragbhayal17/alma-mwc758-zoom512` (account chirag2, pushed 03:58 IST by the watcher script), because of the bash bug in
`results/PROGRESS.md` 2026-10-18 (macOS bash 3.2 has no associative arrays, so every queued job got the last folder, `kaggle/eval_young_test2`). The code it ran is
this folder's `kernel_young_test2.py` (checked by pulling the kernel's source), so the results are valid Young-test2 results. A second, intended run of the same
code (`chiragbhayal/eval-young-test2`, account chirag, pushed 04:18) is a duplicate and serves as a reproducibility check.
