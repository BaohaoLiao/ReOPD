# Data Preparation

This folder will hold the scripts that turn a pool of teacher trajectories into
ReOPD training data.

## Prefix-pool construction (to be added)

ReOPD reuses the teacher's RL rollouts as replayed prefixes. The prep script
fans each teacher trajectory of `K` assistant turns into per-turn training rows
(turn `t` = the student generates turn `t` given the teacher's turns `1..t-1`
and all recorded observations as prompt), and samples positions with the
step-decay schedule `p_t ∝ κ^t` (κ = 0.6 in the paper).

Each output row follows the trainer's input contract (see `train/README.md`):

```json
{"prompt": "...teacher prefix incl. observations...", "label": "...", "metadata": {"task": "retool"}}
```

## Prompt mixing

For multi-environment training, `train/src/prepare_mixed_data.py` combines the
math (ReTool/DAPO) and search (Search-R1 NQ+HotpotQA) prompt files into one
JSONL with `metadata.task` routing. See `train/README.md` for usage.
