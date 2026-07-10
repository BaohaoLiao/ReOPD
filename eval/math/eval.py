from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import time
import importlib
import urllib.parse
from pathlib import Path
from typing import Any
from urllib import request

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SLIME_ROOT = REPO_ROOT / "third_party" / "slime"
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))
# Make sibling modules (eval_generate_with_retool, oat_math_grader, ...)
# importable regardless of the caller's working directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an HF model on AIME-2024 against an already-running SGLang server. "
            "Start the server with eval/math/sglang_serve.sh first."
        )
    )
    parser.add_argument("-n", "--num-samples", type=int, default=1, help="Number of traces to sample per prompt")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum total tokens per trace, counting both model generations and tool observations",
    )
    parser.add_argument(
        "--print-turns",
        action="store_true",
        help="Print intermediate assistant/tool turns for each sampled trace",
    )
    parser.add_argument("--model-path", required=True, help="HF model path used to render chat templates / log in the summary")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path. Defaults to --model-path.",
    )
    parser.add_argument("--dataset", default="baohao/aime24", help="HF dataset name or local dataset path")
    parser.add_argument("--split", default="train", help="Dataset split to evaluate")
    parser.add_argument("--host", default="127.0.0.1", help="SGLang server host")
    parser.add_argument("--port", type=int, default=30000, help="SGLang server port")
    parser.add_argument(
        "--server-wait-timeout",
        type=int,
        default=60,
        help="Seconds to wait for the SGLang server port to accept connections (default: 60).",
    )
    parser.add_argument("--request-timeout", type=int, default=1800, help="HTTP timeout per generation request")
    parser.add_argument("--max-new-tokens", type=int, default=16384, help="Max new tokens per sample")
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=16384,
        help=(
            "Hard cap on input_tokens + max_new_tokens per /generate call. Must match the SGLang "
            "server's --context-length (default 16384). Each turn shrinks max_new_tokens to "
            "max_context_len - len(input_ids) - 16, breaking the trace if no room remains."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    parser.add_argument("--output", default=None, help="Optional JSONL output path for per-example results")
    parser.add_argument("--summary-output", default=None, help="Optional JSON output path for the final summary")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent in-flight requests to the SGLang server (default: 4)",
    )
    parser.add_argument(
        "--sample-timeout",
        type=int,
        default=300,
        help="Max seconds allowed for a single trace (all turns combined). Timed-out traces are scored as incorrect (default: 300).",
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="Print per-turn START/GEN/TOOL debug lines with elapsed times to diagnose timeouts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing --output JSONL: re-run only the timed-out samples "
            "per prompt (keeping non-timeout samples), and re-run all samples for prompts "
            "whose samples were all timed out. The original output file is backed up to "
            "<output>.bak before being overwritten."
        ),
    )
    args = parser.parse_args()
    if args.resume and not args.output:
        parser.error("--resume requires --output to point at the previous run's JSONL file")
    if args.num_samples < 1:
        parser.error("-n/--num-samples must be at least 1")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    return args


def _load_dataset_records(args: argparse.Namespace):
    if Path(args.dataset).exists():
        dataset_path = Path(args.dataset)
        if dataset_path.suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        elif dataset_path.suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")
        else:
            dataset = load_dataset(str(dataset_path), split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    if args.limit is not None:
        limit = min(args.limit, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


# DAPO-style math instruction wrapper. Applied to eval prompts that don't
# already contain it, so the model sees the same prompt format as during
# SFT/RL training.
DAPO_PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: \\boxed{$Answer} where $Answer "
    "is the answer to the problem.\n\n"
)
DAPO_PROMPT_SUFFIX = "\n\nRemember to put your answer on its own line after \"Answer:\"."


def _ensure_dapo_wrap(text: str) -> str:
    """Prepend/append the DAPO math instruction strings if missing."""
    out = text
    if "Solve the following math problem step by step" not in out:
        out = DAPO_PROMPT_PREFIX + out
    if "Remember to put your answer on its own line after" not in out:
        out = out + DAPO_PROMPT_SUFFIX
    return out


def _prompt_to_text(prompt: str | list[dict[str, Any]]) -> str:
    if isinstance(prompt, str):
        return prompt

    lines = []
    for message in prompt:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                else:
                    text_parts.append(str(item))
            content = "\n".join(part for part in text_parts if part)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _load_tokenizer(tokenizer_path: str):
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "transformers is required to run eval/math/eval.py. Install it in the active environment first."
        ) from exc
    return transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _load_math_dapo_compute_score():
    """Return the oat boxed_reward_fn used to grade eval responses.

    Name kept for backwards-compat with call sites; this now returns the
    oat_math_grader grader (boxed extraction + math_equal), not math_dapo.
    """
    module = importlib.import_module("oat_math_grader")
    return module.boxed_reward_fn


def _load_retool_runtime():
    try:
        return importlib.import_module("eval_generate_with_retool")
    except ImportError as exc:
        raise ImportError(
            "Failed to import eval/math/eval_generate_with_retool.py. Check the ReTool dependencies in the active environment."
        ) from exc


def _render_input_ids(tokenizer: Any, prompt: Any) -> list[int]:
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )


async def _post_generate(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/generate"
    return await _post_json_async(url, payload, args.request_timeout)


async def _post_json_async(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Native-async HTTP POST using asyncio streams.

    Unlike asyncio.to_thread(_post_json, ...), this is truly cancellable:
    when asyncio.wait_for times out and raises CancelledError, the TCP
    connection is closed immediately and no thread pool slot is held.
    """
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    body = json.dumps(payload).encode("utf-8")
    http_request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode() + body

    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=10)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"HTTP connect to {host}:{port} timed out after 10s") from exc
    try:
        try:
            writer.write(http_request)
            await asyncio.wait_for(writer.drain(), timeout=10)
            # read(-1) reads until EOF — server closes connection after response with Connection: close
            response_bytes = await asyncio.wait_for(reader.read(-1), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"HTTP request to {host}:{port}{path} timed out after {timeout}s (server overloaded?)"
            ) from exc
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=5)
        except Exception:
            pass

    sep = response_bytes.find(b"\r\n\r\n")
    if sep == -1:
        raise ValueError(f"Malformed HTTP response (no header separator): {response_bytes[:200]}")
    return json.loads(response_bytes[sep + 4:])


def _truncate_text_to_token_budget(tokenizer: Any, text: str, remaining_tokens: int) -> tuple[str, int, bool]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= remaining_tokens:
        return text, len(token_ids), False
    if remaining_tokens <= 0:
        return "", 0, True
    truncated_ids = token_ids[:remaining_tokens]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=False)
    return truncated_text, len(truncated_ids), True


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _read_process_output(*_args: Any, **_kwargs: Any) -> str:  # pragma: no cover - retained for compat
    return ""


def _wait_for_port(host: str, port: int, timeout: int) -> None:
    """Wait for an externally-managed SGLang server to start accepting TCP connections."""
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(
        f"Timed out waiting for SGLang server on {host}:{port} after {timeout}s. "
        f"Is the server running? Last socket error: {last_error}"
    )


def _generate_one(args: argparse.Namespace, tokenizer: Any, prompt: Any) -> str:
    input_ids = _render_input_ids(tokenizer, prompt)
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    url = f"http://{args.host}:{args.port}/generate"
    output = _post_json(url, payload, timeout=args.request_timeout)
    return output["text"]


async def _generate_one_with_tools(
    args: argparse.Namespace,
    tokenizer: Any,
    prompt: Any,
    retool_runtime: Any,
    debug_tag: str = "",
) -> dict[str, Any]:
    tool_registry = retool_runtime.ToolRegistry()
    trace_start = time.time()
    if args.debug_trace:
        print(f"{debug_tag} START backend={tool_registry.backend} session={tool_registry.session_id}", flush=True)
    # Mutable progress tracker so a CancelledError (from outer wait_for timeout)
    # can still report where we stalled.
    progress: dict[str, Any] = {
        "current_turn": -1,
        "current_stage": "init",  # one of: init | gen | tool
        "stage_started_at": trace_start,
    }
    interaction_messages: list[dict[str, Any]] = []
    response_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    tool_call_count = 0
    total_trace_tokens = 0
    stopped_due_to_max_tokens = False
    timed_out = False
    try:
        tool_specs = tool_registry.get_tool_specs()
        max_turns = retool_runtime.TOOL_CONFIGS["max_turns"]
        max_tool_calls = retool_runtime.TOOL_CONFIGS["max_tool_calls"]

        for turn_index in range(max_turns):
            if args.max_tokens is not None and total_trace_tokens >= args.max_tokens:
                stopped_due_to_max_tokens = True
                break

            rendered_prompt = retool_runtime.format_conversation_with_tools(
                prompt=prompt,
                tools=tool_specs,
                messages=interaction_messages,
            )
            input_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
            # Cap max_new_tokens to fit inside the server's context window. SGLang
            # rejects requests where len(input_ids) + max_new_tokens > context_length
            # with HTTP 400 "Requested token count exceeds the model's maximum context length".
            context_budget = max(0, args.max_context_len - len(input_ids) - 16)
            turn_max_new = min(args.max_new_tokens, context_budget)
            if turn_max_new <= 0:
                if args.debug_trace:
                    print(
                        f"{debug_tag} turn={turn_index} CONTEXT-FULL input_tokens={len(input_ids)} "
                        f"max_context_len={args.max_context_len} — stopping trace",
                        flush=True,
                    )
                stopped_due_to_max_tokens = True
                break
            payload = {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": turn_max_new,
                },
            }
            if args.debug_trace:
                print(
                    f"{debug_tag} turn={turn_index} GEN-> input_tokens={len(input_ids)} "
                    f"trace_tokens={total_trace_tokens} elapsed={time.time() - trace_start:.1f}s",
                    flush=True,
                )
            gen_t0 = time.time()
            progress["current_turn"] = turn_index
            progress["current_stage"] = "gen"
            progress["stage_started_at"] = gen_t0
            output = await _post_generate(args, payload)
            gen_dt = time.time() - gen_t0
            cur_response = retool_runtime.postprocess_responses(output["text"])
            if args.debug_trace:
                print(
                    f"{debug_tag} turn={turn_index} GEN<- gen_s={gen_dt:.1f} "
                    f"resp_chars={len(cur_response)}",
                    flush=True,
                )
            if args.max_tokens is not None:
                remaining_tokens = args.max_tokens - total_trace_tokens
                cur_response, response_token_count, response_was_truncated = _truncate_text_to_token_budget(
                    tokenizer,
                    cur_response,
                    remaining_tokens,
                )
            else:
                response_token_count = len(tokenizer(cur_response, add_special_tokens=False)["input_ids"])
                response_was_truncated = False

            response_parts.append(cur_response)
            total_trace_tokens += response_token_count
            turn_record = {
                "turn_index": turn_index,
                "assistant": cur_response,
            }

            if response_was_truncated:
                turn_record["max_tokens_reached"] = True
                turns.append(turn_record)
                stopped_due_to_max_tokens = True
                break

            assistant_message = retool_runtime._build_assistant_message(cur_response)
            if assistant_message is not None:
                interaction_messages.append(assistant_message)

            if args.debug_trace:
                print(
                    f"{debug_tag} turn={turn_index} TOOL-> elapsed={time.time() - trace_start:.1f}s",
                    flush=True,
                )
            tool_t0 = time.time()
            progress["current_stage"] = "tool"
            progress["stage_started_at"] = tool_t0
            next_obs, done, tool_message = await retool_runtime.execute_predictions(cur_response, tool_registry)
            tool_dt = time.time() - tool_t0
            if args.debug_trace:
                print(
                    f"{debug_tag} turn={turn_index} TOOL<- tool_s={tool_dt:.1f} "
                    f"done={done} obs_chars={len(next_obs) if next_obs else 0}",
                    flush=True,
                )
            turn_record["done"] = done
            if done:
                turns.append(turn_record)
                break

            if "<tool_response>" in next_obs:
                tool_call_count += 1

            if args.max_tokens is not None:
                remaining_tokens = args.max_tokens - total_trace_tokens
                next_obs, observation_token_count, observation_was_truncated = _truncate_text_to_token_budget(
                    tokenizer,
                    next_obs,
                    remaining_tokens,
                )
            else:
                observation_token_count = len(tokenizer(next_obs, add_special_tokens=False)["input_ids"])
                observation_was_truncated = False

            response_parts.append(next_obs)
            total_trace_tokens += observation_token_count
            turn_record["tool_observation"] = next_obs
            if tool_message is not None:
                interaction_messages.append(tool_message)
                turn_record["tool_message"] = tool_message.get("content", "")

            if observation_was_truncated:
                turn_record["max_tokens_reached"] = True
                turns.append(turn_record)
                stopped_due_to_max_tokens = True
                break

            turns.append(turn_record)

            if tool_call_count >= max_tool_calls:
                break

    except asyncio.CancelledError:
        # Outer wait_for is cancelling us due to sample_timeout. Don't re-raise:
        # return partial state so the caller can log *where* the trace stalled.
        timed_out = True
        stage = progress.get("current_stage", "init")
        stage_dt = time.time() - progress.get("stage_started_at", trace_start)
        cur_turn = progress.get("current_turn", -1)
        if debug_tag:
            print(
                f"{debug_tag} CANCELLED stage={stage} turn={cur_turn} "
                f"stage_s={stage_dt:.1f} elapsed={time.time() - trace_start:.1f}s "
                f"completed_turns={len(turns)} tool_calls={tool_call_count} "
                f"trace_tokens={total_trace_tokens}",
                flush=True,
            )
    finally:
        try:
            await asyncio.shield(tool_registry.close())
        except Exception:
            pass

    return {
        "response": "".join(response_parts),
        "turns": turns,
        "tool_call_count": tool_call_count,
        "tool_backend": tool_registry.backend,
        "sandbox_session_id": "timeout" if timed_out else tool_registry.session_id,
        "total_trace_tokens": total_trace_tokens,
        "stopped_due_to_max_tokens": stopped_due_to_max_tokens,
        "timed_out": timed_out,
        "timeout_stage": progress.get("current_stage") if timed_out else None,
        "timeout_turn": progress.get("current_turn") if timed_out else None,
    }


def _score_response(grader: Any, prompt: Any, label: str, response: str) -> dict[str, Any]:
    info, score = grader(response, label, fast=False)
    score = float(score)
    return {
        "pred": info.get("pred") if isinstance(info, dict) else None,
        "score": score,
        "acc": score >= 1.0,
        "response": response,
    }


def _print_trace_turns(example_index: int, trace: dict[str, Any]) -> None:
    print(f"\n=== example {example_index} trace {trace['trace_index']} ===")
    print(
        f"backend={trace['tool_backend']} session={trace['sandbox_session_id']} "
        f"tool_calls={trace['tool_call_count']} total_tokens={trace['total_trace_tokens']} "
        f"score={trace['score']:.3f} acc={int(trace['acc'])}"
    )
    for turn in trace.get("turns", []):
        print(f"\n--- turn {turn['turn_index']} assistant ---")
        print(turn.get("assistant", ""))
        if "tool_message" in turn:
            print("\n--- tool message ---")
            print(turn["tool_message"])
        elif "tool_observation" in turn:
            print("\n--- tool observation ---")
            print(turn["tool_observation"])


def _load_resume_records(output_path: Path) -> dict[int, dict[str, Any]]:
    """Load an existing eval JSONL into {example_index: result_record}.

    If multiple lines share the same index (e.g. previous resume runs), the last
    one wins.
    """
    records: dict[int, dict[str, Any]] = {}
    if not output_path.exists():
        return records
    with output_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[resume] skipping malformed line {line_no} in {output_path}: {exc}", flush=True)
                continue
            idx = rec.get("index")
            if isinstance(idx, int):
                records[idx] = rec
    return records


def main() -> None:
    args = parse_args()
    tokenizer = _load_tokenizer(args.tokenizer_path or args.model_path)
    math_dapo_compute_score = _load_math_dapo_compute_score()
    retool_runtime = _load_retool_runtime()
    # Tool specs are static across examples; capture once so we can render the
    # full prompt (system + tools + user) for logging into results.jsonl.
    _tool_specs_for_logging = retool_runtime.ToolRegistry().get_tool_specs()
    dataset = _load_dataset_records(args)
    output_path = Path(args.output) if args.output else None

    resume_records: dict[int, dict[str, Any]] = {}
    if args.resume:
        assert output_path is not None  # guaranteed by parse_args
        resume_records = _load_resume_records(output_path)
        print(
            f"[resume] loaded {len(resume_records)} previous results from {output_path}",
            flush=True,
        )
        if output_path.exists():
            backup_path = output_path.with_suffix(output_path.suffix + ".bak")
            output_path.replace(backup_path)
            print(f"[resume] backed up previous output to {backup_path}", flush=True)

    server_process = None
    try:
        print(
            f"Connecting to SGLang server at http://{args.host}:{args.port} "
            f"(waiting up to {args.server_wait_timeout}s for the port to accept connections)."
        )
        _wait_for_port(args.host, args.port, timeout=args.server_wait_timeout)
        print(f"SGLang server is reachable at http://{args.host}:{args.port}")

        async def _run_eval() -> dict[str, Any]:
            num_examples = len(dataset)
            # One semaphore shared across all examples for the lifetime of the
            # single event loop — avoids the broken-semaphore issue that occurs
            # when asyncio.run() is called once per example.
            semaphore = asyncio.Semaphore(args.max_concurrent)
            output_file = output_path.open("w", encoding="utf-8") if output_path else None
            output_lock = asyncio.Lock()

            # Snapshot rows so we don't iterate the HF dataset across many tasks.
            rows = [(i, dataset[i]) for i in range(num_examples)]

            done_examples = 0
            num_correct = 0.0
            total_score = 0.0
            all_timeout_examples: list[dict[str, Any]] = []
            total_timeouts = 0
            examples_with_any_timeout = 0

            async def _safe_generate(ex_index: int, sample_idx: int, prompt: Any) -> dict[str, Any]:
                async with semaphore:
                    tag = f"[ex{ex_index} s{sample_idx}]"
                    t0 = time.time()
                    try:
                        result = await asyncio.wait_for(
                            _generate_one_with_tools(args, tokenizer, prompt, retool_runtime, debug_tag=tag),
                            timeout=args.sample_timeout,
                        )
                    except asyncio.TimeoutError:
                        # Inner coroutine refused to honor cancellation in time.
                        # This usually means a blocking I/O call we couldn't cancel.
                        print(
                            f"{tag} SAMPLE-TIMEOUT (hard) after {args.sample_timeout}s "
                            f"(wall={time.time() - t0:.1f}s)",
                            flush=True,
                        )
                        return {
                            "response": "",
                            "turns": [],
                            "tool_call_count": 0,
                            "tool_backend": "unknown",
                            "sandbox_session_id": "timeout",
                            "total_trace_tokens": 0,
                            "stopped_due_to_max_tokens": False,
                            "timed_out": True,
                            "timeout_stage": "unknown",
                            "timeout_turn": -1,
                        }
                    except Exception as exc:
                        print(
                            f"{tag} ERROR {type(exc).__name__}: {exc} "
                            f"(wall={time.time() - t0:.1f}s)",
                            flush=True,
                        )
                        return {
                            "response": "",
                            "turns": [],
                            "tool_call_count": 0,
                            "tool_backend": "unknown",
                            "sandbox_session_id": "error",
                            "total_trace_tokens": 0,
                            "stopped_due_to_max_tokens": False,
                            "timed_out": True,
                            "timeout_stage": "error",
                            "timeout_turn": -1,
                        }
                    if result.get("timed_out"):
                        print(
                            f"{tag} SAMPLE-TIMEOUT after {args.sample_timeout}s "
                            f"(wall={time.time() - t0:.1f}s) "
                            f"stage={result.get('timeout_stage')} "
                            f"turn={result.get('timeout_turn')} "
                            f"completed_turns={len(result.get('turns', []))} "
                            f"tool_calls={result.get('tool_call_count', 0)} "
                            f"trace_tokens={result.get('total_trace_tokens', 0)}",
                            flush=True,
                        )
                    return result

            async def _process_example(ex_index: int, row: dict[str, Any]) -> None:
                nonlocal done_examples, num_correct, total_score, total_timeouts, examples_with_any_timeout
                prompt = _ensure_dapo_wrap(row["problem"])
                label = str(row.get("gt", ""))

                # Resume logic: figure out which sample indices need to be (re)run
                # and which existing traces to keep as-is.
                kept_traces: list[dict[str, Any]] = []
                indices_to_run: list[int] = list(range(args.num_samples))
                prev = resume_records.get(ex_index) if args.resume else None
                if prev is not None:
                    prev_traces = prev.get("traces", []) or []
                    prev_by_index = {
                        int(t.get("trace_index", -1)): t for t in prev_traces
                        if isinstance(t.get("trace_index", None), int)
                    }
                    timed_out_indices = {
                        i for i, t in prev_by_index.items() if t.get("timed_out", False)
                    }
                    all_timed_out_prev = (
                        len(prev_by_index) > 0
                        and len(timed_out_indices) == len(prev_by_index)
                    )
                    if all_timed_out_prev:
                        # Retry every sample slot from scratch.
                        indices_to_run = list(range(args.num_samples))
                        kept_traces = []
                    else:
                        # Keep non-timeout traces; rerun timed-out slots plus any
                        # missing slots up to args.num_samples.
                        present_indices = set(prev_by_index.keys())
                        rerun_set = set(timed_out_indices)
                        for i in range(args.num_samples):
                            if i not in present_indices:
                                rerun_set.add(i)
                        indices_to_run = sorted(rerun_set)
                        kept_traces = [
                            prev_by_index[i]
                            for i in sorted(present_indices)
                            if i < args.num_samples and i not in rerun_set
                        ]
                        if not indices_to_run:
                            # Nothing to do for this example — emit the previous record verbatim.
                            async with output_lock:
                                done_examples += 1
                                num_timeouts_prev = int(prev.get("num_timeouts", 0) or 0)
                                num_correct += float(prev.get("avg_acc_excl_timeout", 0.0) or 0.0)
                                total_score += float(prev.get("avg_score_excl_timeout", 0.0) or 0.0)
                                total_timeouts += num_timeouts_prev
                                if num_timeouts_prev > 0:
                                    examples_with_any_timeout += 1
                                if output_file is not None:
                                    output_file.write(json.dumps(prev, ensure_ascii=False) + "\n")
                                    output_file.flush()
                                running_acc = num_correct / done_examples if done_examples else 0.0
                                print(
                                    f"[{done_examples}/{num_examples}] ex{ex_index} "
                                    f"[resume:keep] timeouts={num_timeouts_prev}/{args.num_samples} "
                                    f"running_acc={running_acc:.4f}",
                                    flush=True,
                                )
                            return

                generations = await asyncio.gather(*[
                    _safe_generate(ex_index, i, prompt)
                    for i in indices_to_run
                ])

                new_traces: list[dict[str, Any]] = []
                for sample_idx, generation in zip(indices_to_run, generations):
                    trace = _score_response(math_dapo_compute_score, prompt, label, generation["response"])
                    trace["trace_index"] = sample_idx
                    trace["turns"] = generation["turns"]
                    trace["tool_call_count"] = generation["tool_call_count"]
                    trace["tool_backend"] = generation["tool_backend"]
                    trace["sandbox_session_id"] = generation["sandbox_session_id"]
                    trace["total_trace_tokens"] = generation["total_trace_tokens"]
                    trace["stopped_due_to_max_tokens"] = generation["stopped_due_to_max_tokens"]
                    trace["timed_out"] = generation.get(
                        "timed_out", generation["sandbox_session_id"] in ("timeout", "error")
                    )
                    if trace["timed_out"]:
                        trace["timeout_stage"] = generation.get("timeout_stage")
                        trace["timeout_turn"] = generation.get("timeout_turn")
                    new_traces.append(trace)
                    if args.print_turns:
                        _print_trace_turns(ex_index, trace)

                traces = sorted(
                    kept_traces + new_traces,
                    key=lambda t: int(t.get("trace_index", 0)),
                )

                valid_traces = [t for t in traces if not t["timed_out"]]
                num_timeouts = len(traces) - len(valid_traces)
                all_timed_out = len(valid_traces) == 0

                # avg over all samples (timeouts penalized) — kept for backward compat
                avg_score = sum(t["score"] for t in traces) / len(traces)
                avg_acc = sum(int(t["acc"]) for t in traces) / len(traces)

                # avg over non-timeout samples only — used for the final overall metric
                if all_timed_out:
                    avg_acc_excl_timeout = 0.0
                    avg_score_excl_timeout = 0.0
                else:
                    avg_acc_excl_timeout = sum(int(t["acc"]) for t in valid_traces) / len(valid_traces)
                    avg_score_excl_timeout = sum(t["score"] for t in valid_traces) / len(valid_traces)

                result = {
                    "index": ex_index,
                    "prompt": prompt,
                    "rendered_prompt": retool_runtime.format_conversation_with_tools(
                        prompt=prompt,
                        tools=_tool_specs_for_logging,
                    ),
                    "label": label,
                    "avg_score": avg_score,
                    "avg_acc": avg_acc,
                    "avg_acc_excl_timeout": avg_acc_excl_timeout,
                    "avg_score_excl_timeout": avg_score_excl_timeout,
                    "num_timeouts": num_timeouts,
                    "all_timed_out": all_timed_out,
                    "traces": traces,
                }

                async with output_lock:
                    num_correct += avg_acc_excl_timeout
                    total_score += avg_score_excl_timeout
                    done_examples += 1
                    total_timeouts += num_timeouts
                    if num_timeouts > 0:
                        examples_with_any_timeout += 1
                    if all_timed_out:
                        all_timeout_examples.append({
                            "index": ex_index,
                            "prompt": prompt,
                            "label": label,
                            "num_samples": args.num_samples,
                        })
                    if output_file is not None:
                        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output_file.flush()
                    running_acc = num_correct / done_examples if done_examples else 0.0
                    print(
                        f"[{done_examples}/{num_examples}] ex{ex_index} "
                        f"avg_acc_excl_timeout={avg_acc_excl_timeout:.3f} "
                        f"timeouts={num_timeouts}/{args.num_samples} "
                        f"running_acc={running_acc:.4f}",
                        flush=True,
                    )

            try:
                await asyncio.gather(*[
                    _process_example(idx, row) for idx, row in rows
                ])
            finally:
                if output_file is not None:
                    output_file.close()

            total_samples = num_examples * args.num_samples
            return {
                "num_examples": num_examples,
                "accuracy": num_correct / num_examples if num_examples else 0.0,
                "average_score": total_score / num_examples if num_examples else 0.0,
                "timeout_stats": {
                    "total_samples": total_samples,
                    "total_timeouts": total_timeouts,
                    "timeout_rate": total_timeouts / total_samples if total_samples else 0.0,
                    "examples_with_any_timeout": examples_with_any_timeout,
                    "examples_all_timeout": len(all_timeout_examples),
                    "sample_timeout_seconds": args.sample_timeout,
                    "request_timeout_seconds": args.request_timeout,
                },
                "all_timeout_examples": all_timeout_examples,
                "model_path": args.model_path,
                "dataset": args.dataset,
                "split": args.split,
                "num_samples": args.num_samples,
            }

        summary = asyncio.run(_run_eval())
        print(json.dumps(summary, indent=2))
        if args.summary_output:
            summary_path = Path(args.summary_output)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
            print(f"Summary saved to {summary_path}")
    finally:
        # Externally-managed SGLang server: nothing to tear down here.
        del server_process


if __name__ == "__main__":
    main()