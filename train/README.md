# Retool + Search-R1 Multi-Teacher OPD

This release trains one student model on a mixture of Retool math/code prompts and Search-R1 retrieval QA prompts. Each sample carries `metadata.task`; the rollout, reward, and frozen teacher used for OPD are routed by that task.

- `metadata.task == "retool"` uses Retool-style tool rollouts, Math-DAPO reward, and the Retool teacher.
- `metadata.task == "search-r1"` uses Search-R1 search rollouts, exact-match QA reward, and the Search-R1 teacher.
- The student model is updated by slime/GRPO with OPD teacher logprobs attached in `sample.teacher_log_probs`.

## Repository Layout

```text
train/
  README.md                         # This reproduction guide
  requirements.txt                  # Extra Python dependencies for this release folder
  configs/
    sglang_multiteacher.example.yaml
  scripts/
    run_multiteacher_opd.sh
    run_mixed_two_file_external_teachers_opd.sh
    run_multiteacher_opd_both.sh
    run_multiteacher_opd_v1.sh
  src/
    generate_with_multiteacher_opd.py
    generate_with_retool.py
    generate_with_search.py
    mixed_data_source.py
    prepare_mixed_data.py
    google_search_server.py
    local_search_server.py
    qa_em_format.py
    tool_sandbox.py
```

`scripts/` automatically adds `src/` to `PYTHONPATH`, so the slime custom hook names remain:

```text
generate_with_multiteacher_opd.generate
generate_with_multiteacher_opd.reward_func
generate_with_multiteacher_opd.post_process_rewards
mixed_data_source.MixedTaskDataSource
```

## Environment

This folder assumes Python 3.9+ and the full ReOPD training environment: PyTorch with CUDA, Ray, SGLang, slime, Megatron-LM, and the repo submodules.

From the repo root:

```bash
git submodule update --init --recursive
python -m pip install -r train/requirements.txt
```

If you use the project Dockerfile or an existing training image, install this requirements file on top of that image. It intentionally does not pin CUDA, PyTorch, Ray, SGLang, or Megatron-LM because those should match the cluster image and slime checkout used for training.

## Data Options

You can run with either a pre-combined JSONL file or two original task files that are mixed at rollout time.

### Option A: Pre-combined JSONL

Build one JSONL where every row has `prompt`, `label`, and `metadata.task`:

```bash
python train/src/prepare_mixed_data.py \
  --retool-data /path/to/dapo-math.jsonl \
  --retool-input-key prompt \
  --retool-label-key label \
  --search-data /path/to/nq_hotpotqa_train.parquet \
  --search-input-key prompt \
  --search-label-key reward_model \
  --output /path/to/retool_search_mixed.jsonl
```

The output format is:

```json
{"prompt": "...", "label": "...", "metadata": {"task": "retool"}}
{"prompt": "...", "label": {"ground_truth": ["..."]}, "metadata": {"task": "search-r1"}}
```

Useful flags:

- `--max-retool` and `--max-search` limit either side for smoke tests.
- `--seed` controls the shuffle seed. The default is `42`.
- `--no-shuffle` preserves input order.
- Dotted keys are supported, for example `--search-label-key reward_model.ground_truth`.

### Option B: Two-file Runtime Mixing

Set both task data files and let `mixed_data_source.MixedTaskDataSource` interleave samples during rollout:

```bash
export RETOOL_PROMPT_DATA=/path/to/dapo-math.jsonl
export SEARCH_R1_PROMPT_DATA=/path/to/nq_hotpotqa_train.parquet

export RETOOL_INPUT_KEY=prompt
export RETOOL_LABEL_KEY=label
export SEARCH_R1_INPUT_KEY=prompt
export SEARCH_R1_LABEL_KEY=reward_model

# Repeating a task changes the ratio.
export MIXED_TASK_ORDER=retool,search-r1
```

Examples:

- `MIXED_TASK_ORDER=retool,search-r1` gives roughly 1:1.
- `MIXED_TASK_ORDER=retool,search-r1,search-r1` gives roughly 1:2.
- `MIXED_TASK_ORDER=retool` runs the Retool-only ablation through the same OPD code path.

## Search Backend

Search-R1 rollouts default to local retrieval:

```bash
export SEARCH_URL=http://127.0.0.1:8000/retrieve
```

Multiple local retrievers can be comma-separated. Requests are round-robin load balanced with failover:

```bash
export SEARCH_URL=http://127.0.0.1:8000/retrieve,http://127.0.0.1:8001/retrieve
```

The retriever response is expected to follow the Search-R1 retrieval server format:

```json
{"result": [[{"document": {"contents": "\"Title\"\nDocument text"}}]]}
```

`src/google_search_server.py` is included for the original Search-R1 Google/Serper path, but the released scripts use the local backend by default.

## Teacher Routing

Each task needs a frozen SGLang teacher that accepts `/generate` with `return_logprob=true`.

### External Teacher URLs

Start teacher servers separately and pass their URLs:

```bash
export RETOOL_TEACHER_URL=http://127.0.0.1:13141/generate
export SEARCH_R1_TEACHER_URL=http://127.0.0.1:13142/generate
```

### Slime Multi-model SGLang Config

Alternatively, edit the example config and let slime manage frozen teacher models:

```bash
cp train/configs/sglang_multiteacher.example.yaml \
   train/configs/sglang_multiteacher.yaml

# Edit model_path entries, then:
export SGLANG_CONFIG=/path/to/ReOPD/train/configs/sglang_multiteacher.yaml
```

The default frozen model names are:

- `retool_teacher`
- `search_r1_teacher`

Override them with `RETOOL_TEACHER_MODEL_NAME` and `SEARCH_R1_TEACHER_MODEL_NAME` if your SGLang config uses different names.

You can also provide a JSON map:

```bash
export MULTITEACHER_OPD_TEACHER_URLS='{"retool":"http://host:13141/generate","search-r1":"http://host:13142/generate"}'
```

## Run Training

The most portable entry point is `scripts/run_multiteacher_opd.sh`. It does not start teacher servers for you.

### Pre-combined Data

```bash
export HF_CHECKPOINT=/path/to/student_hf
export REF_LOAD=/path/to/student_torch_dist
export SAVE_DIR=/path/to/output_run
export PROMPT_DATA=/path/to/retool_search_mixed.jsonl

export SEARCH_URL=http://127.0.0.1:8000/retrieve
export RETOOL_TEACHER_URL=http://127.0.0.1:13141/generate
export SEARCH_R1_TEACHER_URL=http://127.0.0.1:13142/generate

bash train/scripts/run_multiteacher_opd.sh
```

### Two-file Runtime Mixing

```bash
export HF_CHECKPOINT=/path/to/student_hf
export REF_LOAD=/path/to/student_torch_dist
export SAVE_DIR=/path/to/output_run

export RETOOL_PROMPT_DATA=/path/to/dapo-math.jsonl
export SEARCH_R1_PROMPT_DATA=/path/to/nq_hotpotqa_train.parquet
export MIXED_TASK_ORDER=retool,search-r1

export SEARCH_URL=http://127.0.0.1:8000/retrieve
export RETOOL_TEACHER_URL=http://127.0.0.1:13141/generate
export SEARCH_R1_TEACHER_URL=http://127.0.0.1:13142/generate

bash train/scripts/run_multiteacher_opd.sh
```

## Script Reference

| Script | Use case | Notes |
| --- | --- | --- |
| `scripts/run_multiteacher_opd.sh` | Recommended portable runner | Supports pre-combined data or two-file runtime mixing. Requires external teachers or `SGLANG_CONFIG`. |
| `scripts/run_mixed_two_file_external_teachers_opd.sh` | Cluster-style 1:1 two-task run with teacher launch | Starts both SGLang teachers, uses two-file data, defaults to task reward enabled. Removes `SAVE_DIR` before running. |
| `scripts/run_multiteacher_opd_both.sh` | Cluster-style two-task OPD run | Starts both teachers with asymmetric GPU defaults, defaults to pure OPD task reward disabled. Kills existing `sglang`, `ray`, and `python` processes and removes `SAVE_DIR`. |
| `scripts/run_multiteacher_opd_v1.sh` | Retool-only ablation through multi-teacher code | Starts only the Retool teacher, sets `MIXED_TASK_ORDER=retool`, defaults to pure OPD. Kills existing `sglang`, `ray`, and `python` processes and removes `SAVE_DIR`. |

Before running the cluster-style scripts, override the default `/root/experiments/...` paths to match your directory layout.

## Important Environment Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `HF_CHECKPOINT` | script-specific | Student Hugging Face checkpoint used by rollout and tokenizer loading. |
| `REF_LOAD` | script-specific | Torch-dist checkpoint loaded by Megatron/slime. |
| `SAVE_DIR` | script-specific | Output directory for checkpoints, logs, rollout state, and debug rollout dumps. |
| `MODEL_CONFIG` | `${SLIME_DIR}/scripts/models/qwen3-4B.sh` | slime model argument file to source. |
| `PROMPT_DATA` | `/root/data/retool_search_mixed.jsonl` in portable script | Pre-combined mixed JSONL input. |
| `RETOOL_PROMPT_DATA` | empty or script-specific | Retool input file for runtime mixing. |
| `SEARCH_R1_PROMPT_DATA` | empty or script-specific | Search-R1 input file for runtime mixing. |
| `MIXED_TASK_ORDER` | `retool,search-r1` | Runtime mixing order and ratio. |
| `SEARCH_URL` | `http://127.0.0.1:8000/retrieve` | Local retrieval endpoint or comma-separated endpoints. |
| `RETOOL_TEACHER_URL` | empty or launched URL | Retool teacher `/generate` endpoint. |
| `SEARCH_R1_TEACHER_URL` | empty or launched URL | Search-R1 teacher `/generate` endpoint. |
| `MULTITEACHER_OPD_USE_TASK_REWARD` | `1` in portable script | `1` trains on task reward plus OPD; `0` logs task reward but returns zero train reward for pure OPD. |
| `MULTITEACHER_OPD_RETOOL_MAX_INFLIGHT` | global teacher limit | Retool teacher request concurrency cap. |
| `MULTITEACHER_OPD_SEARCH_R1_MAX_INFLIGHT` | global teacher limit | Search-R1 teacher request concurrency cap. |
| `MULTITEACHER_OPD_TEACHER_*_TIMEOUT` | see source | Teacher request, connect, socket-read, total-budget, and retry controls. |
| `TOOL_SANDBOX_BACKEND` | `subprocess` | Retool code execution backend. Use `jupyter` only when `jupyter_client` is installed and configured. |
| `TOOL_SANDBOX_CONCURRENCY` | script-specific | Max concurrent Retool code executions. |
| `SEARCH_R1_MAX_TURNS` | `4` | Max Search-R1 tool turns. |
| `GENERATE_SAMPLE_TIMEOUT` | `600` | Wall-clock timeout in seconds for one Retool multi-turn rollout sample. |

## Outputs and Metrics

Training writes to `SAVE_DIR`. The cluster-style scripts also set:

```bash
ROLLOUT_DEBUG_DIR=${SAVE_DIR}/rollout_debug
```

The multi-teacher post-process step logs:

- `rollout/retool_sample_count`
- `rollout/search_r1_sample_count`
- `rollout/retool_task_score_mean`
- `rollout/search_r1_task_score_mean`
- `rollout/retool_train_reward_mean`
- `rollout/search_r1_train_reward_mean`
- `rollout/retool_tool_call_count_mean`

Warnings to watch for:

- `No teacher route configured`: set teacher URLs or `SGLANG_CONFIG`.
- `missing teacher logprobs`: teacher scoring failed for some samples; lower in-flight caps or increase teacher timeouts.
- Search failures from `local_search_server.py`: check `SEARCH_URL` and retriever response format.

## Reproduction Checklist

Record these with every result:

- Git branch and commit SHA.
- Student base checkpoint: `HF_CHECKPOINT` and `REF_LOAD`.
- Retool teacher checkpoint and Search-R1 teacher checkpoint.
- Data files, data key names, and `prepare_mixed_data.py --seed` if using pre-combined data.
- `MIXED_TASK_ORDER`.
- `MULTITEACHER_OPD_USE_TASK_REWARD`.
- GPU allocation for student, rollout engines, teachers, tensor parallelism, and data parallelism.
- `MODEL_CONFIG`, `OPD_KL_COEF`, `KL_LOSS_COEF`, `LR`, `GLOBAL_BATCH_SIZE`, `ROLLOUT_BATCH_SIZE`, and `N_SAMPLES_PER_PROMPT`.
- Search backend endpoint(s) and corpus/version.

## Implementation Notes

- `src/generate_with_multiteacher_opd.py::generate()` routes rollout generation by task.
- `src/generate_with_multiteacher_opd.py::reward_func()` computes the task reward and asks the matching teacher for token logprobs.
- `src/generate_with_multiteacher_opd.py::post_process_rewards()` writes `sample.teacher_log_probs`, preserves GRPO reward normalization, and emits per-task metrics.
- `src/mixed_data_source.py` supports runtime interleaving without building a combined file.
- `src/prepare_mixed_data.py` is the deterministic pre-combine path for release artifacts.

The Retool and Search-R1 task implementations are adapted from their upstream examples; source files retain the upstream references in comments where applicable.
