from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def _read_rows(path: str) -> list[dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required to read parquet input.")
        rows = []
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches():
            rows.extend(batch.to_pylist())
        return rows

    raise ValueError(f"Unsupported file format for {path}. Use .jsonl or .parquet.")


def _resolve_path(data: dict[str, Any], dotted_key: str):
    value = data
    for key in dotted_key.split("."):
        value = value[key]
    return value


def _build_rows(
    rows: list[dict[str, Any]],
    *,
    task: str,
    input_key: str,
    label_key: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    if limit is not None:
        rows = rows[:limit]

    output = []
    for row in rows:
        metadata = row.get("metadata") or {}
        metadata["task"] = task
        output.append(
            {
                "prompt": _resolve_path(row, input_key),
                "label": _resolve_path(row, label_key),
                "metadata": metadata,
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Build a mixed retool/search-r1 jsonl dataset for multi-teacher OPD.")
    parser.add_argument("--retool-data", required=True)
    parser.add_argument("--search-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retool-input-key", default="prompt")
    parser.add_argument("--retool-label-key", default="label")
    parser.add_argument("--search-input-key", default="prompt")
    parser.add_argument("--search-label-key", default="reward_model")
    parser.add_argument("--max-retool", type=int, default=None)
    parser.add_argument("--max-search", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    mixed = []
    mixed.extend(
        _build_rows(
            _read_rows(args.retool_data),
            task="retool",
            input_key=args.retool_input_key,
            label_key=args.retool_label_key,
            limit=args.max_retool,
        )
    )
    mixed.extend(
        _build_rows(
            _read_rows(args.search_data),
            task="search-r1",
            input_key=args.search_input_key,
            label_key=args.search_label_key,
            limit=args.max_search,
        )
    )

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(mixed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(mixed)} rows to {output}")


if __name__ == "__main__":
    main()
