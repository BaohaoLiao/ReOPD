from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import torch

from slime.rollout.data_source import DataSource
from slime.utils.data import Dataset
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


TASK_ALIASES = {
    "retool": "retool",
    "search": "search-r1",
    "search-r1": "search-r1",
    "search_r1": "search-r1",
}

TASK_ENV_PREFIX = {
    "retool": "RETOOL",
    "search-r1": "SEARCH_R1",
}


def _task_from_value(value: str) -> str:
    task = TASK_ALIASES.get(value.strip().lower())
    if task is None:
        raise ValueError(f"Unknown task '{value}'. Expected one of: retool, search-r1.")
    return task


def _env_or_default(name: str, default: str | None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


class MixedTaskDataSource(DataSource):
    """Rollout data source that mixes Retool and Search-R1 samples in one batch.

    The task mix is controlled by MIXED_TASK_ORDER. Repeating a task changes the
    ratio, for example `retool,search-r1,search-r1` gives roughly 1:2.
    """

    def __init__(self, args):
        self.args = args
        self.sample_group_index = 0
        self.sample_index = 0
        self.task_cursor = 0
        self.metadata: dict[str, Any] = {}

        order = os.environ.get("MIXED_TASK_ORDER") or os.environ.get("INTERLEAVED_TASK_ORDER", "retool,search-r1")
        self.task_order = [_task_from_value(item) for item in order.split(",") if item.strip()]
        if not self.task_order:
            raise ValueError("MIXED_TASK_ORDER must contain at least one task.")

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        if (d := args.dump_details) is not None:
            tokenizer.save_pretrained(Path(d) / "tokenizer")
            if processor:
                processor.save_pretrained(Path(d) / "processor")

        self.datasets = {}
        self.sample_offsets = {}
        self.epoch_ids = {}
        self.buffers = {}
        for task in sorted(set(self.task_order)):
            env_prefix = TASK_ENV_PREFIX[task]
            path = os.environ.get(f"{env_prefix}_PROMPT_DATA")
            if not path:
                raise ValueError(
                    f"{env_prefix}_PROMPT_DATA is required for automatic mixed training. "
                    "Set RETOOL_PROMPT_DATA and SEARCH_R1_PROMPT_DATA, or use a pre-combined PROMPT_DATA file."
                )

            dataset = Dataset(
                path,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=_env_or_default(f"{env_prefix}_INPUT_KEY", args.input_key),
                multimodal_keys=args.multimodal_keys,
                label_key=_env_or_default(f"{env_prefix}_LABEL_KEY", args.label_key),
                metadata_key=_env_or_default(f"{env_prefix}_METADATA_KEY", args.metadata_key),
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if args.rollout_shuffle:
                dataset.shuffle(0)

            self.datasets[task] = dataset
            self.sample_offsets[task] = 0
            self.epoch_ids[task] = 0
            self.buffers[task] = []
            logger.info("Loaded %s dataset from %s with %d samples.", task, path, len(dataset))

    def _next_task(self) -> str:
        task = self.task_order[self.task_cursor % len(self.task_order)]
        self.task_cursor += 1
        return task

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        groups = []
        for _ in range(num_samples):
            task = self._next_task()
            buffered = self._get_samples_from_buffer(task, 1)
            if buffered:
                groups.extend(buffered)
            else:
                groups.extend(self._get_samples_from_dataset(task, 1))
        return groups

    def _get_samples_from_buffer(self, task: str, num_samples: int) -> list[list[Sample]]:
        buffer = self.buffers[task]
        num_to_pop = min(len(buffer), num_samples)
        groups = buffer[:num_to_pop]
        del buffer[:num_to_pop]
        return groups

    def _get_samples_from_dataset(self, task: str, num_samples: int) -> list[list[Sample]]:
        dataset = self.datasets[task]
        offset = self.sample_offsets[task]

        if offset + num_samples <= len(dataset):
            prompt_samples = dataset.samples[offset : offset + num_samples]
            self.sample_offsets[task] = offset + num_samples
        else:
            prompt_samples = dataset.samples[offset:]
            remaining = num_samples - len(prompt_samples)
            self.epoch_ids[task] += 1
            if self.args.rollout_shuffle:
                dataset.shuffle(self.epoch_ids[task])
            prompt_samples += dataset.samples[:remaining]
            self.sample_offsets[task] = remaining

        groups = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.metadata = dict(sample.metadata or {})
                sample.metadata["task"] = task
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def add_samples(self, samples: list[list[Sample]]):
        if not samples:
            return
        for group in samples:
            if not group:
                continue
            metadata = group[0].metadata if isinstance(group[0].metadata, dict) else {}
            task = TASK_ALIASES.get(str(metadata.get("task", "")).strip().lower())
            if task is None:
                task = self.task_order[(self.task_cursor - 1) % len(self.task_order)]
            self.buffers[task].append(group)

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        path = os.path.join(self.args.save, f"rollout/mixed_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "sample_offsets": self.sample_offsets,
                "epoch_ids": self.epoch_ids,
                "task_cursor": self.task_cursor,
                "sample_group_index": self.sample_group_index,
                "sample_index": self.sample_index,
                "metadata": self.metadata,
            },
            path,
        )

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset or self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/mixed_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info("Checkpoint %s does not exist.", path)
            return

        state_dict = torch.load(path)
        self.sample_offsets.update(state_dict.get("sample_offsets", {}))
        self.epoch_ids.update(state_dict.get("epoch_ids", {}))
        self.task_cursor = state_dict.get("task_cursor", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_shuffle:
            for task, dataset in self.datasets.items():
                dataset.shuffle(self.epoch_ids[task])

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets.values())
