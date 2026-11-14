# HPC setup and execution

## Verified environment

Runtime root: `$SCRATCH/learning-to-rank`. Interpreter: `$SCRATCH/learning-to-rank/envs/ltr-recovery/bin/python` (Python 3.11.18). Direct package versions observed on HPC are in [retrieval-verified.txt](../../requirements/retrieval-verified.txt). A complete transitive freeze has not been collected into source control.

The original scripts passed on NVIDIA L40S. The refactored encoder and exact-index checks have also passed on HPC. Repeat these checks after subsequent relevant source changes. Do not reinstall or delete an existing environment merely to update project source.

## Transfer from local source

Run in a separate **Windows PowerShell terminal**, from the local project root. These commands copy source files only, not environments or artifacts:

```powershell
$HpcNetId = "YOUR_NETID"
$HpcDestination = "${HpcNetId}@login.torch.hpc.nyu.edu:/scratch/${HpcNetId}/learning-to-rank/"
scp -r src scripts slurm tests requirements $HpcDestination
scp pyproject.toml README.md CONTRIBUTING.md $HpcDestination
```

Transfer a complete source revision in one controlled update. Do not edit or copy source while a job is reading it. Once a remote is configured, prefer a dedicated clean checkout for each recorded experiment. If HPC files were edited directly, reconcile them locally before overwriting them.

## Check wrapper imports before allocating a GPU

Run in the **HPC Bash terminal**:

```bash
cd "$SCRATCH/learning-to-rank"
"$SCRATCH/learning-to-rank/envs/ltr-recovery/bin/python" -I scripts/encoder_smoke_test.py --help
"$SCRATCH/learning-to-rank/envs/ltr-recovery/bin/python" -I scripts/faiss_exact_check.py
```

The second command reuses the existing encoder_vectors.npz artifact. Missing artifacts require generating embeddings first. Script wrappers add only this checkout's src directory to sys.path because Python -I removes automatic script-directory imports.

## GPU regression

Run from the source root. Account selection is explicit on the submission command; the template does not embed a billing account:

```bash
# Replace this value with an allocation authorized for the project.
export SLURM_ACCOUNT="YOUR_SLURM_ACCOUNT"
mkdir -p logs
sbatch --account="$SLURM_ACCOUNT" slurm/smoke.sbatch gpu
sbatch --account="$SLURM_ACCOUNT" slurm/smoke.sbatch encoder
```

These short jobs are independent. Inspect their named log files for successful completion. After the encoder job passes, rerun faiss_exact_check.py to validate newly generated vectors. Do not assume completion solely because sbatch returned a job ID. The template uses SLURM_SUBMIT_DIR for source and LTR_PROJECT_ROOT or SCRATCH for artifacts. LTR_SOURCE_ROOT can explicitly select another source checkout. Do not specify a partition manually for this workflow.

## Model download

If the snapshot is missing, run download_encoder.py on the login node before requesting a GPU. The versioned encoder.json revision must match an existing model manifest; resolve intentional revision changes explicitly. No login credential is required for this public checkpoint. Inference is offline and never follows a moving model branch.

## Dependency installation for a fresh environment

Existing project installations do not need this step. Keep caches and temporary files under scratch and target the explicit project interpreter:

```bash
mkdir -p "$SCRATCH/learning-to-rank/cache/pip" "$SCRATCH/learning-to-rank/cache/tmp"
TMPDIR="$SCRATCH/learning-to-rank/cache/tmp" \
"$SCRATCH/learning-to-rank/envs/ltr-recovery/bin/python" -I -m pip --isolated install \
  --cache-dir "$SCRATCH/learning-to-rank/cache/pip" \
  --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1"
TMPDIR="$SCRATCH/learning-to-rank/cache/tmp" \
"$SCRATCH/learning-to-rank/envs/ltr-recovery/bin/python" -I -m pip --isolated install \
  --cache-dir "$SCRATCH/learning-to-rank/cache/pip" \
  --index-url https://pypi.org/simple \
  -c requirements/bootstrap-constraints.txt -r requirements/retrieval.in
```

Direct retrieval dependencies are now pinned to the observed versions, including Transformers 5.22.2. Transitive dependencies are not yet locked; capture a new freeze after any solve. Constraints do not install a missing package: include numpy==2.2.6 when setting up an otherwise empty environment. A full environment lock and container build are future work.

## Prior failures and operational limits

Home reached its file-count quota despite low byte usage. Conda attempted writes under home even with a scratch package cache. The first ltr environment had package records but no executable payload. A fresh ltr-recovery prefix and cache with copy mode, disabled registration, and no external plugins produced working Python and pip. Multiple factors changed; the exact original cause was not proven. Keep unrelated environments intact; do not update shared Conda base or globally redefine HOME.

Use myquota and my_slurm_accounts for current account information. Scratch is working storage, not a backup. Save source and important outputs durably. A logout does not normally remove scratch files but does reset shell exports; the wrappers use explicit paths and cache initialization to reduce dependence on shell state. Never train or run corpus benchmarks on login nodes.

## Official references

- https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/
- https://services.rt.nyu.edu/docs/hpc/tools_and_software/utils/
- https://services.rt.nyu.edu/docs/hpc/storage/intro_and_data_management/
- https://pytorch.org/get-started/previous-versions/

## Authentication retry and FAISS fallback diagnostics

Select keyboard-interactive authentication if SSH does not present the device login flow:

```powershell
$HpcNetId = "YOUR_NETID"
ssh -o PreferredAuthentications=keyboard-interactive -o PasswordAuthentication=no "${HpcNetId}@login.torch.hpc.nyu.edu"
```

Complete device authentication privately; do not include codes in logs or source. FAISS may log missing AVX-specific modules before successfully loading its generic library. If the final load and correctness checks pass, a reinstall is not required solely for those informational messages. Record actual CPU implementation and host context for future performance benchmarks rather than assuming this fallback is equally fast.
