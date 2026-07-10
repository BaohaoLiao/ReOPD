from __future__ import annotations

import asyncio

# Keep the stdlib loop policy used by the existing OPD helpers. It is more
# stable with many concurrent subprocess/tool and aiohttp calls.
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import json
import os
import sys
from typing import Any

import aiohttp
import torch

import generate_with_retool
import generate_with_search
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


TASK_ALIASES = {
    "retool": "retool",
    "search": "search-r1",
    "search-r1": "search-r1",
    "search_r1": "search-r1",
}

TEACHER_MODEL_ENV = {
    "retool": "RETOOL_TEACHER_MODEL_NAME",
    "search-r1": "SEARCH_R1_TEACHER_MODEL_NAME",
}

TEACHER_URL_ENV = {
    "retool": "RETOOL_TEACHER_URL",
    "search-r1": "SEARCH_R1_TEACHER_URL",
}

DEFAULT_TEACHER_MODEL = {
    "retool": "retool_teacher",
    "search-r1": "search_r1_teacher",
}

_session_lock = asyncio.Lock()
_session_by_loop: dict[int, aiohttp.ClientSession] = {}
# Per-(loop, task) in-flight semaphore. The two teachers have very different
# capacities (e.g. retool on 3-4 GPUs, search-r1 on 1 GPU). A single shared
# semaphore lets the slow single-GPU teacher hold slots that the retool teacher
# needs, and lets up to MAX_INFLIGHT heavy retool prefills hit a 3-GPU teacher
# at once -> KV overrun -> server queueing -> sock_read timeout -> retries.
# Bounding each teacher independently keeps each server's queue shallow.
_semaphore_by_loop: dict[tuple[int, str], asyncio.Semaphore] = {}
_TEACHER_MAX_INFLIGHT = int(os.environ.get("MULTITEACHER_OPD_TEACHER_MAX_INFLIGHT", "64"))
_TEACHER_MAX_INFLIGHT_BY_TASK = {
    "retool": int(
        os.environ.get("MULTITEACHER_OPD_RETOOL_MAX_INFLIGHT", str(_TEACHER_MAX_INFLIGHT))
    ),
    "search-r1": int(
        os.environ.get("MULTITEACHER_OPD_SEARCH_R1_MAX_INFLIGHT", str(_TEACHER_MAX_INFLIGHT))
    ),
}
_TEACHER_REQUEST_TIMEOUT = int(os.environ.get("MULTITEACHER_OPD_TEACHER_REQUEST_TIMEOUT", "600"))
_TEACHER_CONNECT_TIMEOUT = int(os.environ.get("MULTITEACHER_OPD_TEACHER_CONNECT_TIMEOUT", "60"))
_TEACHER_SOCK_READ_TIMEOUT = int(os.environ.get("MULTITEACHER_OPD_TEACHER_SOCK_READ_TIMEOUT", "300"))
_TEACHER_TOTAL_BUDGET = int(os.environ.get("MULTITEACHER_OPD_TEACHER_TOTAL_BUDGET", "900"))
_TEACHER_MAX_RETRIES = int(os.environ.get("MULTITEACHER_OPD_TEACHER_MAX_RETRIES", "2"))
_LAST_TASK_REWARD_MEAN = {"retool": 0.0, "search-r1": 0.0}
_LAST_TASK_TRAIN_REWARD_MEAN = {"retool": 0.0, "search-r1": 0.0}
_LAST_TASK_SCORE_MEAN = {"retool": 0.0, "search-r1": 0.0}
_LAST_RETOOL_TOOL_CALL_COUNT_MEAN = 0.0


def _metadata(sample: Sample) -> dict[str, Any]:
    return sample.metadata if isinstance(sample.metadata, dict) else {}


def _get_task(sample: Sample) -> str:
    metadata = _metadata(sample)
    task = metadata.get("task") or metadata.get("source") or metadata.get("dataset")
    task = TASK_ALIASES.get(str(task).strip().lower() if task is not None else "")
    if task is None:
        raise ValueError(
            "Each sample must set metadata.task to 'retool' or 'search-r1'. "
            "Build the mixed dataset with src/prepare_mixed_data.py or provide an equivalent metadata column."
        )
    return task


def _use_task_reward() -> bool:
    return os.environ.get("MULTITEACHER_OPD_USE_TASK_REWARD", "1").lower() not in {"0", "false", "no"}


def _get_teacher_url(args, task: str) -> str:
    url = os.environ.get(TEACHER_URL_ENV[task])
    if url:
        return url

    url_map = os.environ.get("MULTITEACHER_OPD_TEACHER_URLS")
    if url_map:
        urls = json.loads(url_map)
        if task in urls:
            return urls[task]

    model_name = os.environ.get(TEACHER_MODEL_ENV[task], DEFAULT_TEACHER_MODEL[task])
    routers = getattr(args, "sglang_model_routers", None) or {}
    if model_name in routers:
        return get_model_url(args, model_name, "/generate")

    raise ValueError(
        f"No teacher route configured for task '{task}'. Set {TEACHER_URL_ENV[task]} to an external "
        f"SGLang /generate URL, set MULTITEACHER_OPD_TEACHER_URLS, or define model '{model_name}' "
        "in --sglang-config with update_weights: false."
    )


def _get_semaphore(task: str) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = (id(loop), task)
    sem = _semaphore_by_loop.get(key)
    if sem is None:
        limit = _TEACHER_MAX_INFLIGHT_BY_TASK.get(task, _TEACHER_MAX_INFLIGHT)
        sem = asyncio.Semaphore(limit)
        _semaphore_by_loop[key] = sem
    return sem


async def _get_session() -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    key = id(loop)
    sess = _session_by_loop.get(key)
    if sess is not None and not sess.closed:
        return sess

    async with _session_lock:
        sess = _session_by_loop.get(key)
        if sess is not None and not sess.closed:
            return sess

        timeout = aiohttp.ClientTimeout(
            total=_TEACHER_REQUEST_TIMEOUT,
            connect=_TEACHER_CONNECT_TIMEOUT,
            sock_connect=_TEACHER_CONNECT_TIMEOUT,
            sock_read=_TEACHER_SOCK_READ_TIMEOUT,
        )
        connector = aiohttp.TCPConnector(
            limit=128,
            limit_per_host=128,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
        )
        sess = aiohttp.ClientSession(trust_env=False, timeout=timeout, connector=connector)
        _session_by_loop[key] = sess
        return sess


async def generate(args, sample: Sample, sampling_params) -> Sample:
    task = _get_task(sample)
    if task == "retool":
        return await generate_with_retool.generate(args, sample, sampling_params)
    if task == "search-r1":
        return await generate_with_search.generate(args, sample, sampling_params)
    raise AssertionError(f"Unhandled task: {task}")


async def _teacher_logprob(args, sample: Sample, task: str) -> dict[str, Any] | None:
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        payload["image_data"] = [
            encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]
        ]

    last_err: Exception | None = None
    sem = _get_semaphore(task)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TEACHER_TOTAL_BUDGET
    url = _get_teacher_url(args, task)

    for attempt in range(_TEACHER_MAX_RETRIES):
        if deadline - loop.time() <= 0:
            break
        try:
            session = await _get_session()
            async with sem:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = e
            await asyncio.sleep(min(2**attempt, 8))
        except RuntimeError as e:
            msg = str(e)
            if "File descriptor" not in msg and "transport" not in msg:
                raise
            last_err = e
            await asyncio.sleep(0.05 * (attempt + 1))

    print(
        f"[multiteacher-opd] WARN: teacher /generate failed for task={task} "
        f"after {_TEACHER_MAX_RETRIES} retries (url={url}, tokens={len(sample.tokens)}): {last_err!r}",
        file=sys.stderr,
        flush=True,
    )
    return None


async def _task_reward(args, sample: Sample, task: str) -> float:
    if task == "retool":
        solution_str = generate_with_retool._prompt_to_text(sample.prompt) + sample.response
        ground_truth = sample.label if sample.label is not None else ""
        reward = generate_with_retool.math_dapo_compute_score(
            solution_str,
            ground_truth,
            strict_box_verify=True,
        )
        task_score = float(reward["score"])
        num_turns = int(getattr(sample, "tool_call_count", 0) or 0)
        shaped_reward = task_score
        if shaped_reward < 0:
            tool_call_reward = (num_turns - 2) / 2 * 0.1
            shaped_reward = min(-0.6, shaped_reward + tool_call_reward)
        sample._opd_task_score_unshaped = task_score  # type: ignore[attr-defined]
        return float(shaped_reward)
    elif task == "search-r1":
        reward = await generate_with_search.reward_func(args, sample)
    else:
        raise AssertionError(f"Unhandled task: {task}")

    if isinstance(reward, dict):
        if "score" in reward:
            scalar_reward = float(reward["score"])
            sample._opd_task_score_unshaped = scalar_reward  # type: ignore[attr-defined]
            return scalar_reward
        if "reward" in reward:
            scalar_reward = float(reward["reward"])
            sample._opd_task_score_unshaped = scalar_reward  # type: ignore[attr-defined]
            return scalar_reward
        raise ValueError(f"Cannot extract scalar reward from dict keys: {sorted(reward.keys())}")
    scalar_reward = float(reward)
    sample._opd_task_score_unshaped = scalar_reward  # type: ignore[attr-defined]
    return scalar_reward


async def _reward_one(args, sample: Sample) -> float:
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    task = _get_task(sample)
    # Keep teacher scoring and task scoring in one coroutine. This mirrors the
    # Retool-only OPD path more closely than spawning a per-sample gather.
    teacher_response = await _teacher_logprob(args, sample, task)
    task_reward = await _task_reward(args, sample, task)

    sample._opd_task = task  # type: ignore[attr-defined]
    sample._opd_task_score = task_reward  # type: ignore[attr-defined]
    sample._opd_teacher_response = teacher_response  # type: ignore[attr-defined]
    return task_reward


async def reward_func(args, sample, **kwargs):
    if isinstance(sample, list):
        return await asyncio.gather(*[_reward_one(args, item) for item in sample])
    return await _reward_one(args, sample)


def _extract_teacher_log_probs(sample: Sample) -> torch.Tensor:
    response_length = sample.response_length
    resp = getattr(sample, "_opd_teacher_response", None)
    if resp is None:
        return torch.zeros(response_length, dtype=torch.float32)

    full = torch.tensor(
        [item[0] for item in resp["meta_info"]["input_token_logprobs"][1:]],
        dtype=torch.float32,
    )
    sliced = full[:0] if response_length == 0 else full[-response_length:]
    if sliced.numel() != response_length:
        pad = torch.zeros(response_length - sliced.numel(), dtype=torch.float32)
        sliced = torch.cat([pad, sliced])
    return sliced


def _normalize_rewards(args, rewards: list[float]) -> list[float]:
    if not _use_task_reward():
        return [0.0] * len(rewards)

    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        reward_tensor = torch.tensor(rewards, dtype=torch.float)
        expected = args.n_samples_per_prompt * args.rollout_batch_size
        if reward_tensor.shape[-1] == expected:
            reward_tensor = reward_tensor.reshape(-1, args.n_samples_per_prompt)
        else:
            reward_tensor = reward_tensor.view(-1, reward_tensor.shape[-1])
        reward_tensor = reward_tensor - reward_tensor.mean(dim=-1, keepdim=True)

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            reward_tensor = reward_tensor / (reward_tensor.std(dim=-1, keepdim=True) + 1e-6)

        return reward_tensor.flatten().tolist()

    return rewards


def post_process_rewards(args, samples: list[Sample], **kwargs):
    global _LAST_RETOOL_TOOL_CALL_COUNT_MEAN

    raw_rewards = []
    sample_tasks = []
    task_counts = {"retool": 0, "search-r1": 0}
    task_raw_rewards = {"retool": [], "search-r1": []}
    task_scores = {"retool": [], "search-r1": []}
    retool_tool_call_counts = []
    failed_teachers = {"retool": 0, "search-r1": 0}
    has_round_number = any(sample.metadata and "round_number" in sample.metadata for sample in samples)

    for sample in samples:
        task = getattr(sample, "_opd_task", None) or _get_task(sample)
        sample_tasks.append(task)
        task_counts[task] += 1
        if getattr(sample, "_opd_teacher_response", None) is None:
            failed_teachers[task] += 1

        sample.teacher_log_probs = _extract_teacher_log_probs(sample)
        sample.metadata = _metadata(sample)
        sample.metadata["task"] = task
        if has_round_number:
            sample.metadata.setdefault("round_number", 0)

        raw_reward = float(getattr(sample, "_opd_task_score", 0.0) or 0.0)
        raw_rewards.append(raw_reward)
        task_raw_rewards[task].append(raw_reward)
        task_score = float(getattr(sample, "_opd_task_score_unshaped", raw_reward) or 0.0)
        task_scores[task].append(task_score)
        if task == "retool":
            retool_tool_call_counts.append(float(getattr(sample, "tool_call_count", 0) or 0))

    if any(failed_teachers.values()):
        print(
            f"[multiteacher-opd] WARN: missing teacher logprobs: {failed_teachers}",
            file=sys.stderr,
            flush=True,
        )

    train_rewards = _normalize_rewards(args, raw_rewards)
    task_train_rewards = {"retool": [], "search-r1": []}
    for task, reward in zip(sample_tasks, train_rewards, strict=False):
        task_train_rewards[task].append(float(reward))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def mean_or_previous(task: str, values: list[float], cache: dict[str, float]) -> float:
        if values:
            cache[task] = mean(values)
        return cache[task]

    retool_reward_mean = mean_or_previous("retool", task_raw_rewards["retool"], _LAST_TASK_REWARD_MEAN)
    search_r1_reward_mean = mean_or_previous("search-r1", task_raw_rewards["search-r1"], _LAST_TASK_REWARD_MEAN)
    retool_task_score_mean = mean_or_previous("retool", task_scores["retool"], _LAST_TASK_SCORE_MEAN)
    search_r1_task_score_mean = mean_or_previous("search-r1", task_scores["search-r1"], _LAST_TASK_SCORE_MEAN)
    if retool_tool_call_counts:
        _LAST_RETOOL_TOOL_CALL_COUNT_MEAN = mean(retool_tool_call_counts)
    retool_tool_call_count_mean = _LAST_RETOOL_TOOL_CALL_COUNT_MEAN
    retool_train_reward_mean = mean_or_previous(
        "retool", task_train_rewards["retool"], _LAST_TASK_TRAIN_REWARD_MEAN
    )
    search_r1_train_reward_mean = mean_or_previous(
        "search-r1", task_train_rewards["search-r1"], _LAST_TASK_TRAIN_REWARD_MEAN
    )

    log_metrics = {
        "rollout/student_task_score": mean(task_scores["retool"] + task_scores["search-r1"]),
        "rollout/shaped_reward_mean": mean(raw_rewards),
        "rollout/retool_count": task_counts["retool"],
        "rollout/search_r1_count": task_counts["search-r1"],
        "rollout/retool_sample_count": task_counts["retool"],
        "rollout/search_r1_sample_count": task_counts["search-r1"],
        "rollout/retool_reward_mean": retool_reward_mean,
        "rollout/search_r1_reward_mean": search_r1_reward_mean,
        "rollout/retool_shaped_reward_mean": retool_reward_mean,
        "rollout/retool_task_score_mean": retool_task_score_mean,
        "rollout/search_r1_task_score_mean": search_r1_task_score_mean,
        "rollout/retool_tool_call_count_mean": retool_tool_call_count_mean,
        "rollout/retool_train_reward_mean": retool_train_reward_mean,
        "rollout/search_r1_train_reward_mean": search_r1_train_reward_mean,
    }
    print(
        "[multiteacher-opd] "
        f"retool_samples={task_counts['retool']} "
        f"search_r1_samples={task_counts['search-r1']} "
        f"retool_reward_mean={log_metrics['rollout/retool_reward_mean']:.4f} "
        f"retool_task_score_mean={log_metrics['rollout/retool_task_score_mean']:.4f} "
        f"retool_tool_calls_mean={log_metrics['rollout/retool_tool_call_count_mean']:.2f} "
        f"search_r1_reward_mean={log_metrics['rollout/search_r1_reward_mean']:.4f} "
        f"retool_train_reward_mean={log_metrics['rollout/retool_train_reward_mean']:.4f} "
        f"search_r1_train_reward_mean={log_metrics['rollout/search_r1_train_reward_mean']:.4f}",
        flush=True,
    )

    try:
        import wandb

        if wandb.run is not None and raw_rewards:
            wandb.log(log_metrics)
    except ImportError:
        pass

    return raw_rewards, train_rewards
