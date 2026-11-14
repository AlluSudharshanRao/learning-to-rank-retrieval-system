#!/usr/bin/env bash
# Submit from the repository root with an explicit permitted --account.
#SBATCH --job-name=ltr-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

runtime_root="${LTR_PROJECT_ROOT:-${SCRATCH:?SCRATCH is not set}/learning-to-rank}"
source_root="${LTR_SOURCE_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch}}"
python_path="$runtime_root/envs/ltr-recovery/bin/python"

case "${1:-encoder}" in
  encoder) script="encoder_smoke_test.py" ;;
  gpu) script="gpu_smoke_test.py" ;;
  *) echo "Usage: sbatch --account=ACCOUNT slurm/smoke.sbatch [encoder|gpu]" >&2; exit 2 ;;
esac

test -x "$python_path"
test -f "$source_root/scripts/$script"
cd "$source_root"
exec "$python_path" -I "scripts/$script" --project-root "$runtime_root"
