# Git and HPC workflow

## Source and runtime storage

Maintain source in a local checkout and execute GPU workloads on HPC. Reconcile any changes made directly on HPC before updating that checkout. Record the source commit and worktree status for each experiment.

Runtime storage defaults to `$SCRATCH/learning-to-rank`. The [HPC runbook](hpc_setup.md) describes source transfers and job execution. Source transfers include both `src/` and `scripts/`; datasets, environments, and generated artifacts remain in runtime storage.

## Commit workflow

From the repository root:

```bash
git status --short
git diff
git add README.md CONTRIBUTING.md pyproject.toml docs src scripts tests slurm requirements .github .gitignore .gitattributes
git diff --cached --stat
git diff --cached
git commit -m "Describe the change"
```

Commit cohesive changes with their tests and documentation. Inspect staged files for generated outputs and credentials. Git-ignored local notes remain outside the repository. `.gitattributes` enforces LF line endings for Python, shell, and text files used on Linux.

## Remote synchronization

Configure the intended repository as the remote before the first push. On HPC, check for uncommitted changes before pulling or replacing source. Use a clean checkout for recorded experiments, and avoid modifying source while a job is reading it.

Large datasets, model checkpoints, caches, and indexes use separate artifact storage. Keep only small reviewed fixtures and selected results in Git. A local checkout is not a remote backup.
