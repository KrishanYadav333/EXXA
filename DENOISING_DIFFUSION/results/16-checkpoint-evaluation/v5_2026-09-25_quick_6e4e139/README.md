# 16-checkpoint-evaluation, first Kaggle run (QUICK), 2026-09-25

Kaggle Version 5 (number from the author, RULES.md #10; it was `v_pending_...` until then). Cell 0b pulled `6e4e139`
(size fix for notebook 14's 600 px arms live). Cells: the Kaggle copy had the old full-profile cell 6; the user pasted an override cell below it
(`PROFILE='quick'`, `LE_LIMIT=2`, `INCLUDE` regex), so this run is 34 checkpoints x 3 cases = **102 rows in 49 min**
(measured per-row seconds: 256 px U-Net ~21, stack_kin ~33, sg ~24, 600 px ~74, 320/480 px ~13 to 17 on the SG cube).

Cases: `run_0002_00560_rt_00`, `run_0002_00560_rt_01` (2 of the 5 line-emission holdout cubes) and `sg_v2`. **Every line-emission mean below is over 2 cubes, not the
5 in the 05 tables: not comparable to them.** sd is across cubes, not seeds.

**Failure (cosmetic, after all scoring):** cell 20 `AttributeError: 'NoneType' object has no attribute 'group'`. A figure sheet's name matched none of the kinds in the
case-name regex, so the display loop raised, and Run All stopped before cell 22 (`collect_outputs`). Scoring, tables and the 84 figures (26 per case + 6 dashboards)
were all written to `/kaggle/working`, so they are in the notebook Output, but the run was not collected into a run folder. Fixed: unmatched names are skipped
and the regex knows the calibration / integrated-spectrum / error-hist / ensemble kinds.

`run_log.txt` is the notebook's printed text. **The CSV `nb16_eval_rows.csv` and the PNGs are not in this folder yet**: they are in the Kaggle Output and have to be downloaded.
Until then `resid_err_ratio`, the wiggle-figure ordering and the invented-structure detail are unavailable to the record.
