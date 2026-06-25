#!/usr/bin/env bash
set -euo pipefail

cd /opt/lean-treepo

methods=("grpo" "grpo_tree_sampling" "treepo_fixed_init" "treepo_more_init")
if [[ -n "${EXPERIMENT:-}" ]]; then
  methods=("${EXPERIMENT}")
fi

for method in "${methods[@]}"; do
  marker="/workspace/.lean_treepo_${method}_success"
  if [[ "${FORCE_RERUN:-false}" != "true" && -f "${marker}" ]]; then
    echo "Skipping ${method}; marker exists at ${marker}."
    continue
  fi

  echo "Starting ${method}."
  if [[ "${USE_ACCELERATE:-false}" == "true" ]]; then
    accelerate launch \
      --num_processes "${ACCELERATE_NUM_PROCESSES:-1}" \
      -m lean_treepo.train --method "${method}" "$@"
  else
    python -m lean_treepo.train --method "${method}" "$@"
  fi

  mkdir -p "$(dirname "${marker}")"
  {
    echo "completed_at=$(date -Iseconds)"
    echo "method=${method}"
    echo "output_root=${OUTPUT_ROOT:-/workspace/outputs/treepo-runs}"
  } > "${marker}"
done
