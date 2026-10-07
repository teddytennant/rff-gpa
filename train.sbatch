#!/usr/bin/env bash
#
# Smoke run for RFF-GPA (arXiv 2610.08578). Copied from templates/sbatch.sh
# with the two placeholders filled in.
#
# __VENV_SETUP__   : reuse the shared CUDA jax venv
# __RUN_COMMAND__  : 20 SGD steps of the MC ELBO on 1x H200

#SBATCH --job-name=arxiv-impl-rff-gpa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -uo pipefail
cd "$(dirname "$0")"
ROOT="${SLURM_SUBMIT_DIR:-/work/ttennant1/arxiv-impl/20261007T102107Z}"
exec > "$ROOT/rff-gpa.out" 2> "$ROOT/rff-gpa.err"

PY="$HOME/arxiv-jax/venv/bin/python"
export JAX_PLATFORMS=cuda
unset JAX_PLATFORM_NAME || true
export PYTHONPATH="$ROOT/rff-gpa${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/rff-gpa"

"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
"$PY" smoke_gpu.py
rc=$?

echo "exit $rc"
exit $rc
