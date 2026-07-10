# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import json
import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Import tool sandbox functionality
from eval_tool_sandbox import SEMAPHORE, TOOL_CONFIGS, ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use Python "
    "tools to solve mathematical problems. When you need "
    "to perform calculations, use the code_interpreter "
    "tool to execute code and get results."
)

TOOL_SYSTEM_PROMPT = """# Tools

You may call one function at a time to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
__TOOLS__
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
After a tool is executed, you will receive the tool result in a user message wrapped in <tool_response></tool_response> tags.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def format_conversation_with_tools(
    prompt: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] = None,
    system_prompt: str = None,
    messages: list[dict[str, Any]] = None,
) -> str:
    """Format conversation using a static system prompt and explicit message rendering."""

    # Prepare messages
    messages_to_render = []

    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = DEFAULT_SYSTEM_PROMPT

    if tools:
        tool_lines = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
        system_content = f"{system_content}\n\n{TOOL_SYSTEM_PROMPT.replace('__TOOLS__', tool_lines)}"

    messages_to_render.append({"role": "system", "content": system_content})

    prompt_messages = _normalize_prompt_messages(prompt)
    if prompt_messages:
        messages_to_render.extend(prompt_messages)

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(
            message
            for message in messages
            if _should_keep_message(message)
        )

    rendered_parts = []
    for message in messages_to_render:
        role = message["role"]
        if role == "system":
            rendered_parts.append(f"<|im_start|>system\n{message['content']}<|im_end|>")
        elif role == "user":
            rendered_parts.append(f"<|im_start|>user\n{message['content']}<|im_end|>")
        elif role == "assistant":
            assistant_text = ["<|im_start|>assistant", message.get("content", "")]
            for tool_call in message.get("tool_calls", []):
                assistant_text.append("<tool_call>")
                assistant_text.append(json.dumps(tool_call["function"], ensure_ascii=False))
                assistant_text.append("</tool_call>")
            assistant_text.append("<|im_end|>")
            rendered_parts.append("\n".join(part for part in assistant_text if part != ""))
        elif role == "tool":
            rendered_parts.append(
                "<|im_start|>user\n"
                "<tool_response>\n"
                f"{message['content']}\n"
                "</tool_response><|im_end|>"
            )

    rendered_parts.append("<|im_start|>assistant\n")
    return "\n".join(rendered_parts)


def _render_recorded_messages(messages: list[dict[str, Any]]) -> str:
    """Render recorded chat messages into transcript text without adding a new assistant turn."""
    rendered_parts = []
    for message in messages:
        role = message["role"]
        if role == "system":
            rendered_parts.append(f"<|im_start|>system\n{message['content']}<|im_end|>")
        elif role == "user":
            rendered_parts.append(f"<|im_start|>user\n{message['content']}<|im_end|>")
        elif role == "assistant":
            assistant_text = ["<|im_start|>assistant", message.get("content", "")]
            for tool_call in message.get("tool_calls", []):
                assistant_text.append("<tool_call>")
                assistant_text.append(json.dumps(tool_call["function"], ensure_ascii=False))
                assistant_text.append("</tool_call>")
            assistant_text.append("<|im_end|>")
            rendered_parts.append("\n".join(part for part in assistant_text if part != ""))
        elif role == "tool":
            rendered_parts.append(
                "<|im_start|>user\n"
                "<tool_response>\n"
                f"{message['content']}\n"
                "</tool_response><|im_end|>"
            )
    return "\n".join(rendered_parts)


def _stringify_message_content(content: Any) -> str:
    """Convert structured message content into plain text for prompting and rewards."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(_stringify_message_content(item.get("content")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "content" in content:
            return _stringify_message_content(content.get("content"))
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _has_message_content(content: Any) -> bool:
    """Return whether a message content should be kept in the rendered chat."""
    if isinstance(content, str):
        return bool(content.strip())
    return bool(_stringify_message_content(content).strip())


def _should_keep_message(message: Any) -> bool:
    """Return whether a structured message should be kept in the chat history."""
    if not isinstance(message, dict):
        return False
    if message.get("tool_calls"):
        return True
    return _has_message_content(message.get("content", ""))


def _extract_first_tool_call(prediction: str) -> dict[str, Any] | None:
    """Extract the first tool call from a model response."""
    tool_call_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if not tool_call_match:
        return None

    try:
        json_str = tool_call_match.group(1).replace("\n", "\\n")
        tool_call_data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(tool_call_data, dict):
        return None
    return tool_call_data


def _build_assistant_message(prediction: str) -> dict[str, Any] | None:
    """Build a structured assistant message from a model response."""
    sanitized_prediction = postprocess_responses(prediction)
    tool_call = _extract_first_tool_call(sanitized_prediction)
    if tool_call is None:
        content = sanitized_prediction.rstrip()
        if not content:
            return None
        return {"role": "assistant", "content": content}

    tool_call_pattern = r"<tool_call>\s*.*?\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, sanitized_prediction, re.DOTALL)
    if tool_call_match is None:
        return {"role": "assistant", "content": sanitized_prediction.rstrip()}

    content = sanitized_prediction[: tool_call_match.start()].rstrip()
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": tool_call}],
    }


def _render_tool_message(tool_message: dict[str, Any]) -> str:
    """Render a structured tool message into the rollout transcript format."""
    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\n"
        f"{tool_message['content']}\n"
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_initial_recorded_messages(
    prompt: str | list[dict[str, Any]], system_prompt: str = None
) -> list[dict[str, Any]]:
    """Build the initial structured message list used for debugging and replay."""
    recorded_messages = [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}]
    recorded_messages.extend(_normalize_prompt_messages(prompt))
    return recorded_messages


def _trim_response_token_prefix(
    tokenizer,
    raw_token_ids: list[int],
    raw_log_probs: list[float],
    sanitized_response: str,
    *,
    sandbox_session_id: str | None = None,
    turn: int | None = None,
) -> tuple[list[int], list[float]]:
    """Trim generated token/logprob sequences to the sanitized response prefix."""
    raw_response = tokenizer.decode(raw_token_ids)
    if sanitized_response == raw_response:
        return raw_token_ids, raw_log_probs

    if not sanitized_response:
        return [], []

    for prefix_len in range(1, len(raw_token_ids) + 1):
        if tokenizer.decode(raw_token_ids[:prefix_len]) == sanitized_response:
            return raw_token_ids[:prefix_len], raw_log_probs[:prefix_len]

    sanitized_token_ids = tokenizer(sanitized_response, add_special_tokens=False)["input_ids"]
    raw_suffix = raw_response[-200:].replace("\n", "\\n")
    sanitized_suffix = sanitized_response[-200:].replace("\n", "\\n")
    print(
        "[token-trim-mismatch] "
        f"session={sandbox_session_id} turn={turn} "
        f"raw_tokens={len(raw_token_ids)} sanitized_tokens={len(sanitized_token_ids)} "
        f"raw_chars={len(raw_response)} sanitized_chars={len(sanitized_response)} "
        f"raw_suffix={raw_suffix!r} sanitized_suffix={sanitized_suffix!r}"
    )
    trimmed_log_probs = raw_log_probs[: len(sanitized_token_ids)]
    if len(trimmed_log_probs) < len(sanitized_token_ids):
        trimmed_log_probs = trimmed_log_probs + [0.0] * (len(sanitized_token_ids) - len(trimmed_log_probs))
    return sanitized_token_ids, trimmed_log_probs


def _apply_final_token_clip(
    tokenizer,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    loss_masks: list[int],
    rollout_log_probs: list[float] | None,
    max_context_length: int,
) -> tuple[list[int], str, list[int], list[int], list[float] | None, bool]:
    """Hard-clip the final sample to the max token budget used by training."""
    max_response_tokens = max(0, max_context_length - len(prompt_token_ids))
    was_clipped = len(response_token_ids) > max_response_tokens

    if not was_clipped:
        response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        return prompt_token_ids + response_token_ids, response_text, response_token_ids, loss_masks, rollout_log_probs, False

    clipped_response_token_ids = response_token_ids[:max_response_tokens]
    clipped_loss_masks = loss_masks[:max_response_tokens]
    clipped_log_probs = rollout_log_probs[:max_response_tokens] if rollout_log_probs is not None else None
    clipped_response = tokenizer.decode(clipped_response_token_ids, skip_special_tokens=False)
    clipped_tokens = prompt_token_ids + clipped_response_token_ids
    return (
        clipped_tokens,
        clipped_response,
        clipped_response_token_ids,
        clipped_loss_masks,
        clipped_log_probs,
        True,
    )


def _normalize_prompt_messages(prompt: str | list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Normalize prompt input into chat messages."""
    if prompt is None:
        return []
    if isinstance(prompt, str) and "<|im_start|>" in prompt:
        parsed_messages = _parse_chat_transcript(prompt)
        if parsed_messages:
            return parsed_messages
    if isinstance(prompt, list):
        normalized_messages = []
        for message in prompt:
            if not isinstance(message, dict):
                content = str(message)
                if not content.strip():
                    continue
                normalized_messages.append({"role": "user", "content": content})
                continue
            content = _stringify_message_content(message.get("content", ""))
            if not content.strip():
                continue
            normalized_messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": content,
                }
            )
        return normalized_messages
    content = str(prompt)
    if not content.strip():
        return []
    return [{"role": "user", "content": content}]


def _parse_chat_transcript(prompt: str) -> list[dict[str, str]]:
    """Parse a rendered chat transcript back into structured messages."""
    pattern = re.compile(r"<\|im_start\|>(system|user|assistant)\n?(.*?)(?=<\|im_end\|>)<\|im_end\|>", re.DOTALL)
    messages: list[dict[str, str]] = []

    for role, content in pattern.findall(prompt):
        text = content.strip()
        if role == "assistant" and not text:
            # Ignore generation prompts like <|im_start|>assistant\n with no content.
            continue
        if not text:
            continue
        messages.append({"role": role, "content": text})

    return messages


def _prompt_to_text(prompt: str | list[dict[str, Any]]) -> str:
    """Convert prompt data into a string for reward computation."""
    if isinstance(prompt, str):
        return prompt
    normalized_messages = _normalize_prompt_messages(prompt)
    lines = []
    for message in normalized_messages:
        lines.append(f"{message['role']}: {message['content']}")
    return "\n".join(lines)


def _find_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """Find all top-level ``\\boxed{...}`` spans with balanced braces.

    Returns a list of ``(start, end, inner_content)`` tuples where ``start`` is
    the index of the leading backslash and ``end`` is the index just past the
    closing brace. Handles arbitrary brace nesting (e.g. ``\\frac{a^{b}}{c}``
    inside the box) which the previous regex could not.
    """
    spans: list[tuple[int, int, str]] = []
    needle = "\\boxed{"
    i = 0
    while True:
        start = text.find(needle, i)
        if start == -1:
            break
        # Walk forward from the opening brace, tracking depth.
        j = start + len(needle)
        depth = 1
        content_start = j
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "\\" and j + 1 < len(text):
                # Skip escaped character (e.g. \{ or \}) so it does not affect depth.
                j += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, j + 1, text[content_start:j]))
                    j += 1
                    break
            j += 1
        if depth != 0:
            # Unbalanced; stop searching to avoid false matches.
            break
        i = j
    return spans


def postprocess_predictions(prediction: str):
    """Extract action and content from prediction string"""
    # Stop on any assistant content that contains a boxed final answer.
    # Prefer the last boxed span so trailing reasoning before the final box is tolerated.
    boxed_spans = _find_boxed_spans(prediction)
    if boxed_spans:
        content = boxed_spans[-1][2].strip()
        return "answer", content

    # Then check for <tool_call> tags (new format from Jinja2 template)
    tool_call_data = _extract_first_tool_call(prediction)
    if tool_call_data:
        tool_name = tool_call_data.get("name")
        arguments = tool_call_data.get("arguments", {})

        if tool_name == "code_interpreter":
            code = arguments.get("code", "")
            if code.strip():
                return "code", code

    # Then check for <code> tags
    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content

    # Finally check for ```python code blocks (lowest priority)
    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    # Handle <tool_call> tags (new format from Jinja2 template)
    if "<tool_call>" in resp:
        # Keep only the first complete <tool_call>...</tool_call> block.
        tool_call_pattern = r"<tool_call>\s*.*?\s*</tool_call>"
        match = re.search(tool_call_pattern, resp, re.DOTALL)
        if match:
            return resp[: match.end()]

    # Handle <code> tags
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Stop once any boxed final answer appears in the assistant response.
    if "\\boxed{" in resp:
        boxed_spans = _find_boxed_spans(resp)
        if boxed_spans:
            return resp[: boxed_spans[-1][1]]

    return resp


async def execute_predictions(
    prediction: str, tool_registry: ToolRegistry
) -> tuple[str, bool, dict[str, Any] | None]:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Content is already the Python code (extracted by
        # postprocess_predictions)
        code = content.strip()
        if code:
            result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            tool_message = {"role": "tool", "content": str(result)}
            next_obs = _render_tool_message(tool_message)
            done = False
        else:
            tool_message = {"role": "tool", "content": "Error: No Python code found"}
            next_obs = _render_tool_message(tool_message)
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
        tool_message = None
    else:
        tool_message = {
            "role": "tool",
            "content": (
                "The previous action is invalid. "
                "If executing code, you should return a JSON object inside <tool_call></tool_call>. "
                "If giving the final answer, you should use the format 'Answer: \\boxed{answer}'. PLease try again."
            ),
        }
        next_obs = _render_tool_message(tool_message)
        done = False

    return next_obs, done, tool_message


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    state = GenerateState(args)
    tool_registry = ToolRegistry()
    sandbox_session_id = tool_registry.session_id
    sandbox_backend = tool_registry.backend
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Set up the initial prompt with system prompt and tools (outside the loop)
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)
    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # Build recorded_messages with the full system prompt (including tool descriptions)
    # so that payload_text faithfully reflects what the model actually sees.
    full_system_prompt = DEFAULT_SYSTEM_PROMPT
    if tool_specs:
        tool_lines = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tool_specs)
        full_system_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{TOOL_SYSTEM_PROMPT.replace('__TOOLS__', tool_lines)}"
    recorded_messages = _build_initial_recorded_messages(sample.prompt, system_prompt=full_system_prompt)
    interaction_messages: list[dict[str, Any]] = []
    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu
    
    max_context_length = max_context_length - 16  # Leave some buffer tokens for safety

    # Keep the sample structurally valid even if a later rollout turn aborts.
    sample.tokens = list(prompt_tokens_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.sandbox_session_id = sandbox_session_id
    sample.sandbox_backend = sandbox_backend
    sample.messages = list(recorded_messages)

    print(f"[sandbox] session={sandbox_session_id} backend={sandbox_backend} sample_start")

    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds
    last_finish_reason = None

    for turn in range(TOOL_CONFIGS["max_turns"]):
        prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs, messages=interaction_messages)
        current_prompt_token_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # Check if total length exceeds max context length
        total_length = len(current_prompt_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break
        remaining_context = max_context_length - total_length

        # Use token IDs instead of text
        current_token_ids = current_prompt_token_ids
        current_sampling_params = dict(sampling_params)
        current_sampling_params["max_new_tokens"] = min(
            current_sampling_params["max_new_tokens"],
            remaining_context,
        )
        if current_sampling_params["max_new_tokens"] <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": current_sampling_params,
            "return_logprob": True,  # Request log probabilities for training
        }

        print(
            f"[sandbox] session={sandbox_session_id} backend={sandbox_backend} turn={turn} tool_calls={tool_call_count}"
        )

        output = await post(url, payload)

        last_finish_reason = output["meta_info"]["finish_reason"]["type"]

        # Handle abort
        if last_finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            break

        if "output_token_logprobs" in output["meta_info"]:
            raw_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            raw_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            raw_response = state.tokenizer.decode(raw_response_token_ids)
            cur_response = postprocess_responses(raw_response)
            cur_response_token_ids, cur_log_probs = _trim_response_token_prefix(
                state.tokenizer,
                raw_response_token_ids,
                raw_log_probs,
                cur_response,
                sandbox_session_id=sandbox_session_id,
                turn=turn,
            )
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs

        else:
            cur_response = output["text"]
            cur_response = postprocess_responses(cur_response)
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        assistant_message = _build_assistant_message(cur_response)
        if assistant_message is not None:
            interaction_messages.append(assistant_message)
            recorded_messages.append(assistant_message)
            sample.messages = list(recorded_messages)

        # Check length limit
        if last_finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
            break

        next_obs, done, tool_message = await execute_predictions(cur_response, tool_registry)
        if done:
            break

        # Count tool calls (when we get interpreter output, it means a tool
        # was called)
        if "<tool_response>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        remaining_observation_tokens = max_context_length - (
            len(current_prompt_token_ids) + len(cur_response_token_ids)
        )
        if remaining_observation_tokens <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        if len(obs_tokens_ids) > remaining_observation_tokens:
            obs_tokens_ids = obs_tokens_ids[-remaining_observation_tokens:]
            next_obs = state.tokenizer.decode(obs_tokens_ids, skip_special_tokens=False)
            sample.status = Sample.Status.TRUNCATED
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        if tool_message is not None:
            interaction_messages.append(tool_message)
            recorded_messages.append(tool_message)
            sample.messages = list(recorded_messages)

        # Add dummy log probs for observation tokens (they won't be used due to loss_mask=0)
        # Check if maximum tool call count reached
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

            assert len(response_token_ids) == len(
                sample.rollout_log_probs
            ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

        if sample.status == Sample.Status.TRUNCATED:
            break

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    # Set sample attributes
    (
        final_tokens,
        final_response,
        final_response_token_ids,
        final_loss_masks,
        final_rollout_log_probs,
        final_was_clipped,
    ) = _apply_final_token_clip(
        state.tokenizer,
        prompt_tokens_ids,
        response_token_ids,
        loss_masks,
        sample.rollout_log_probs,
        max_context_length,
    )

    sample.tokens = final_tokens
    sample.response_length = len(final_response_token_ids)
    sample.response = final_response
    sample.loss_mask = final_loss_masks
    sample.rollout_log_probs = final_rollout_log_probs
    if final_was_clipped:
        sample.status = Sample.Status.TRUNCATED

    # Store payload information for wandb logging
    sample.payload_text = _render_recorded_messages(recorded_messages)
    sample.payload_has_system = "<|im_start|>system" in sample.payload_text
    sample.payload_has_tools = "# Tools" in sample.payload_text
    sample.sandbox_session_id = sandbox_session_id
    sample.sandbox_backend = sandbox_backend

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count

    # Expose round_number in metadata so slime logs it under multi_turn_metric/
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["round_number"] = tool_call_count

    # Set status
    if sample.status not in {Sample.Status.TRUNCATED, Sample.Status.ABORTED}:
        match last_finish_reason:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

    await tool_registry.close()
    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string
    solution_str = _prompt_to_text(sample.prompt) + sample.response

    # Get ground truth answer - label is a string, not a dict
    ground_truth = sample.label if sample.label is not None else ""

    # Get tool call count as num_turns
    num_turns = getattr(sample, "tool_call_count", 0)

    # use \\boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
