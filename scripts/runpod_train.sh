#!/usr/bin/env bash
set -euo pipefail

cd /opt/lean-treepo

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for model/dataset access and optional Hub pushes." >&2
  exit 2
fi

if [[ -z "${KIMINA_BASE_URL:-}" ]]; then
  echo "KIMINA_BASE_URL is required for Lean verifier rewards." >&2
  exit 2
fi

if [[ "${REPORT_TO:-none}" == *"wandb"* && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required when REPORT_TO includes wandb." >&2
  exit 2
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ONCE_MARKER="${RUN_ONCE_MARKER:-/workspace/.lean_treepo_success}"
IDLE_AFTER_SUCCESS="${IDLE_AFTER_SUCCESS:-true}"
FORCE_RERUN="${FORCE_RERUN:-false}"

idle_forever() {
  echo "Container will stay alive. Stop the RunPod pod when you are done to release the GPU."
  tail -f /dev/null
}

if [[ "${FORCE_RERUN}" != "true" && -f "${RUN_ONCE_MARKER}" ]]; then
  echo "Found success marker at ${RUN_ONCE_MARKER}; skipping training to avoid a RunPod restart loop."
  echo "Set FORCE_RERUN=true or delete the marker to train again."
  if [[ "${IDLE_AFTER_SUCCESS}" == "true" ]]; then
    idle_forever
  fi
  exit 0
fi

if [[ "${RUN_ALL_EXPERIMENTS:-false}" == "true" ]]; then
  scripts/run_all_experiments.sh "$@"
elif [[ "${USE_ACCELERATE:-false}" == "true" ]]; then
  accelerate launch \
    --num_processes "${ACCELERATE_NUM_PROCESSES:-1}" \
    -m lean_treepo.train --method "${METHOD:-grpo}" "$@"
else
  python -m lean_treepo.train --method "${METHOD:-grpo}" "$@"
fi

mkdir -p "$(dirname "${RUN_ONCE_MARKER}")"
{
  echo "completed_at=$(date -Iseconds)"
  echo "method=${METHOD:-grpo}"
  echo "hub_model_prefix=${HUB_MODEL_PREFIX:-}"
  echo "output_root=${OUTPUT_ROOT:-/workspace/outputs/treepo-runs}"
  echo "kimina_base_url=${KIMINA_BASE_URL:-}"
  echo "image=${HOSTNAME:-unknown}"
} > "${RUN_ONCE_MARKER}"
echo "Training completed successfully; wrote success marker to ${RUN_ONCE_MARKER}."

if [[ "${IDLE_AFTER_SUCCESS}" == "true" ]]; then
  idle_forever
fi
