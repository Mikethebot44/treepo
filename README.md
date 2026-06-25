# Lean TreePO

RunPod-ready TRL scaffold for comparing GRPO and TreePO-style post-training on Lean theorem proving. It trains `Pythagoras-LM/Pythagoras-Prover-4B` on deterministic subsets of `AI-MO/NuminaMath-LEAN` and scores generated Lean with a hosted Kimina Lean Server.

## Methods

- `grpo`: stock TRL `GRPOTrainer` with independent rollouts.
- `grpo_tree_sampling`: GRPO objective with segmented TreePO rollouts.
- `treepo_fixed_init`: TreePO rollouts plus segment-level tree advantages with fixed initial divergence.
- `treepo_more_init`: TreePO rollouts plus segment-level tree advantages with random initial divergence from 2 to 8.

The model receives no tools during generation. Kimina is used only after generation to compute verifier rewards.

## Local Smoke

```bash
python -m pip install -e ".[dev]"
python -m compileall src tests
python -m pytest -q
$env:KIMINA_BASE_URL="http://localhost:8000"; python -m lean_treepo.train --help
```

Tiny training smoke, assuming a reachable Kimina server:

```bash
$env:HF_TOKEN="hf_xxx"
$env:KIMINA_BASE_URL="https://your-kimina-server"
$env:REPORT_TO="none"
python -m lean_treepo.train --method grpo --train-size 8 --val-size 0 --test-size 0 --max-steps 1 --num-generations 2 --push-to-hub false
```

## RunPod

Build and push an immutable image:

```bash
docker build -t docker.io/<user>/lean-treepo:v0.1.0 .
docker push docker.io/<user>/lean-treepo:v0.1.0
docker buildx imagetools inspect docker.io/<user>/lean-treepo:v0.1.0
```

Use these RunPod template settings:

- Image: `docker.io/<user>/lean-treepo:v0.1.0`
- Container disk: 80 GB or larger
- Volume mount path: `/workspace`
- Volume disk: enough for checkpoints and Hugging Face cache
- Start command: leave default entrypoint
- Secrets/env: copy `configs/runpod.env.example` and set real `HF_TOKEN`, `WANDB_API_KEY`, `KIMINA_BASE_URL`, optional `KIMINA_API_KEY`

Run one method:

```bash
METHOD=treepo_fixed_init
```

Run all four methods sequentially:

```bash
RUN_ALL_EXPERIMENTS=true
```

Each method writes outputs under `/workspace/outputs/treepo-runs/<method>`. The entrypoint writes success markers under `/workspace` to avoid restart loops.

## Production Defaults

The env example keeps smoke-safe data and step counts. The algorithm defaults match the replication plan:

- `NUM_GENERATIONS=16`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_COMPLETION_LENGTH=7168`
- `TREE_SEGMENT_LENGTH=512`
- `TREE_MAX_DEPTH=14`
- `TEMPERATURE=1.0`
- `TOP_P=1.0`
- `BETA=0.0`
- `LOSS_TYPE=dapo`
- `SCALE_REWARDS=group`

LoRA defaults are consistent across ablations: rank 64, alpha 128, dropout 0.05, bf16, optional 4-bit QLoRA, and Qwen projection/MLP target modules.

## Evaluation And Plots

```bash
python -m lean_treepo.evaluate --model-or-adapter your-hf-user/pythagoras-prover-4b-numina-lean-treepo-fixed-init
python -m lean_treepo.plots --runs-dir /workspace/outputs/treepo-runs
```

Evaluation reports pass@k and majority@k by sampling completions, extracting Lean, and verifying with Kimina.
# treepo
