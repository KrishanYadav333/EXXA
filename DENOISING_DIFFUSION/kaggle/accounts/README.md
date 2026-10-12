# Running jobs from several Kaggle accounts

Why: each account has its own weekly GPU quota and 2 concurrent sessions. Kaggle's terms restrict multiple accounts / credential sharing
(re-read them; a suspension is possible and falls on the account holder). Use only accounts whose owners agreed.

## One-time setup (account owners, on their own machines or yours, never in chat)
1. `mkdir ~/.kaggle-<name> && chmod 700 ~/.kaggle-<name>`; put that account's `kaggle.json` there; `chmod 600` it. `<name>` is a label (e.g. backup, friend1).
2. Test: `kaggle/accounts/kaggle_as.sh <name> kernels list --mine | head -3`.

## Data access (website only; the API has no sharing endpoint)
On kaggle.com, for each of these private items owned by the main account, add the other account as a viewer/collaborator:
datasets `line-emission-data`, `exxa-sg-synth-pairs`, `self-gravitating-v2`, `exxa-wiggle-code`, and a Speedie dataset (see below).
Then a kernel on the other account can list `krishanyadav333/<dataset>` in `dataset_sources` as usual.

Kernel-output inputs (`kernel_sources`) DO cross accounts when the source kernel is public or shared with the target (nb14 resume across accounts worked
this way, Krishan 2026-10-11); otherwise they do not. "Add Input" has no version selector
(memory: two-kaggle-accounts). So jobs that run on another account should not depend on another account's kernel output:
- Speedie cubes (`speedie-prep-fast` output): make them a dataset (kernel page -> Output -> New Dataset) owned by the main account, then share it.
- Training jobs end with their own eval (line holdouts, 9074, analytic, SG v2, Speedie), so no checkpoint has to move between accounts
  (RULES.md #3: a checkpoint is never uploaded as a dataset).

## Per-job
Copy a kernel folder, set `"id": "<username>/<slug>"` in `kernel-metadata.json`, keep `dataset_sources` as `krishanyadav333/...`, then
`kaggle_as.sh <name> kernels push -p <folder>`; fetch logs and outputs with the same account's key. Log which account ran what in RUNS.md.

## Automation (orchestrator.py)
`python kaggle/accounts/orchestrator.py accounts | submit DIR --slug NAME [--account X] [--dry-run] | status | collect [--out DIR]`.
It finds accounts from the key files, tries them in order until one accepts the push (2 concurrent GPU sessions each; a "weekly GPU quota" reply
parks that account until the next Saturday 05:30 IST), polls status, and downloads output + log of finished jobs. State is in state.json (no secrets,
git-ignored). Only the dataset / kernel sharing on kaggle.com stays manual. Quota left is not exposed by the API: it is inferred from the push replies.

## No website sharing needed: each account gets its own copy
With an account's key (key file as above) the data can be uploaded under that account directly:
`sync_account.py data <name>` (~11.8 GB: line-emission-data 8.1, exxa-sg-synth-pairs 2.0, self-gravitating-v2 1.7), `sync_account.py code <name>` (flat files + MANIFEST),
then `orchestrator.py submit kaggle/speedie_prep --slug speedie-prep-fast --account <name>` (CPU, no GPU quota, ~1 h) so the Speedie cubes exist in that account.
`orchestrator.py submit` rewrites the owner prefix of dataset_sources / kernel_sources to the target account. Upload time is set by your uplink
(about 20 min at 10 MB/s, 100 min at 2 MB/s, per account). Website sharing is the alternative if you prefer not to upload copies.

## Cross-account kernel outputs (confirmed by use)
A notebook's saved output can be attached as an input to its copy on another account when the source is public or shared (nb14 resume, 2026-10-11).
So: `orchestrator.py submit kaggle/speedie_prep --slug speedie-prep-fast --public` once, then jobs on other accounts use `--shared-kernel-sources`;
the Speedie cubes are derived from the CC-BY Zenodo release, so public is fine. Mentor data (line emission, sg_synth, v2) stays private: per-account copies.

## Chosen route (2026-10-11): private sharing, no uploads
On kaggle.com, share these from the main account with each other account (viewer is enough): datasets `line-emission-data`, `exxa-sg-synth-pairs`,
`self-gravitating-v2`, `exxa-wiggle-code`, and the kernel `speedie-prep-fast`. Verify: `orchestrator.py check` (each account must see every item).
Submit with `--shared-datasets --shared-kernel-sources`: sources stay owned by the main account, so there is one copy of the data and one Speedie run.
