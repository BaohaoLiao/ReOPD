def extract_boxed_answer(prediction: str) -> str | None:
    """Extracts the answer from any occurrence of \\boxed{...} (LaTeX style)."""
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, prediction)
    if matches:
        return matches[-1].strip()
    return None
# Adapted from search-r1/generate_with_search.py.
# This is the eval-only variant: we strip the slime-specific `generate()` /
# `reward_func()` entry points so the file can be imported by `eval.py`
# without needing the slime training stack.
import asyncio
import json
import re
from typing import Any

# Configuration for Search-R1 eval. Override SEARCH_R1_CONFIGS["search_backend"],
# SEARCH_R1_CONFIGS["local"]["search_url"], etc. before calling search().

# System prompt and tool definition (retool style)
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools to answer questions. "
    "You have access to a search tool. When you need to look up information, "
    "call the search tool as described below."
)

TOOL_SYSTEM_PROMPT = """# Tools

You may call one function at a time to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
<tool name=\"search\" description=\"Search for information.\" args=\"{\\\"query\\\": string}\" />
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
After a tool is executed, you will receive the tool result in a user message wrapped in <tool_response></tool_response> tags.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

SEARCH_R1_CONFIGS = {
    "max_turns": 4,
    "topk": 3,
    "search_concurrency": 256,
    "search_backend": "local",
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # Comma-separate multiple local retriever URLs.
        "proxy": None,
    },
    "google": {
        "api_key": "your_api_key_here",
        "snippet_only": True,
        "proxy": None,
    },
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])



def _passages2string(retrieval_result):
    """Convert retrieval results to a formatted reference string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference



async def search(query: str) -> str:
    """Perform search using either local search engine or Google search."""
    backend = SEARCH_R1_CONFIGS["search_backend"]
    if backend == "local":
        from local_search_server import local_search
        local_config = SEARCH_R1_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        from google_search_server import google_search
        google_config = SEARCH_R1_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    else:
        raise ValueError(f"Unknown search backend: {backend}. Must be either 'local' or 'google'.")
    return _passages2string(result)



def postprocess_responses(resp: str) -> str:
    """Truncate the assistant response at the first closing </tool_call> or </answer> tag."""
    if "</tool_call>" in resp:
        return resp.split("</tool_call>")[0] + "</tool_call>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    return resp



# Tool call extraction (retool style)
def _extract_first_tool_call(prediction: str) -> dict[str, Any] | None:
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




def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    """Return (action, content) where action is 'search'|'answer'|None."""
    tool_call = _extract_first_tool_call(prediction)
    if tool_call and tool_call.get("name") == "search":
        return "search", tool_call["arguments"]["query"]
    # fallback: check for <answer>
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        return "answer", match.group(1).strip()
    # fallback: check for \\boxed{...}
    boxed = extract_boxed_answer(prediction)
    if boxed is not None:
        return "boxed_answer", boxed
    return None, ""



async def execute_predictions(prediction: str) -> tuple[str, bool]:
    """Run the action implied by `prediction`. Return (next_observation, done)."""
    action, content = postprocess_predictions(prediction)
    if action == "search":
        try:
            async with SEMAPHORE:
                search_results = await search(content)
            tool_response_content = search_results.strip()
        except Exception as e:
            tool_response_content = f"[ERROR] {type(e).__name__}: {e}"
        # Return as <tool_response> in user message, retool style
        next_obs = (
            "<|im_start|>user\n<tool_response>\n"
            f"{tool_response_content}\n"
            "</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )
        return next_obs, False
    if action == "answer" or action == "boxed_answer":
        return "", True
    # Move the invalid action message into a user message, matching retool style
    next_obs = (
        "<|im_start|>user\n"
        "Your previous action is invalid. "
        "If you want to use a tool, you should return a <tool_call>...</tool_call> block. "
        "If you want to give the final answer, you should put the answer in \\boxed{{}}. "
        "Please try again.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    return next_obs, False
# ...existing code...

# Conversation rendering (retool style)
def format_conversation_with_tools(
    prompt: str | list[dict[str, Any]],
    system_prompt: str = None,
    messages: list[dict[str, Any]] = None,
) -> str:
    """Format conversation using a static system prompt and explicit message rendering."""
    messages_to_render = []
    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = DEFAULT_SYSTEM_PROMPT
    system_content = f"{system_content}\n\n{TOOL_SYSTEM_PROMPT}"
    messages_to_render.append({"role": "system", "content": system_content})
    # Add user prompt
    if isinstance(prompt, str):
        messages_to_render.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        messages_to_render.extend(prompt)
    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)
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
                "<|im_start|>user\n<tool_response>\n"
                f"{message['content']}\n"
                "</tool_response><|im_end|>"
            )
    rendered_parts.append("<|im_start|>assistant\n")
    return "\n".join(rendered_parts)
