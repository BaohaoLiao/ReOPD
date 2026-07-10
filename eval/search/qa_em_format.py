# Adapt from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/verl/utils/reward_score/qa_em_format.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import json
import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    tool_valid, tool_reason = is_valid_tool_sequence(text)
    if tool_valid:
        return True, tool_reason

    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)

    if not assistant_match:
        return False, "Missing assistant marker"

    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]

    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # Now check for proper sequence pattern and no extraneous content

    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end

    # 3. Check each part
    for _i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue

        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def _tool_response_contents(text: str) -> list[str]:
    pattern = r"<tool_response>\s*(.*?)\s*</tool_response>"
    return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]


def _extract_tool_calls(text: str) -> list[dict]:
    calls = []
    for payload in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        try:
            call = json.loads(payload.replace("\n", "\\n"))
        except Exception:
            continue
        if isinstance(call, dict):
            calls.append(call)
    return calls


def is_valid_tool_sequence(text):
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    if not assistant_match:
        return False, "Missing assistant marker"

    content = text[assistant_match.end():]
    tool_open = len(re.findall(r"<tool_call>", content))
    tool_close = len(re.findall(r"</tool_call>", content))
    response_open = len(re.findall(r"<tool_response>", content))
    response_close = len(re.findall(r"</tool_response>", content))
    if tool_open != tool_close:
        return False, f"Mismatch in tool_call tags: {tool_open} opening vs {tool_close} closing tags"
    if response_open != response_close:
        return False, f"Mismatch in tool_response tags: {response_open} opening vs {response_close} closing tags"

    tool_calls = _extract_tool_calls(content)
    if len(tool_calls) != tool_open:
        return False, "Invalid tool_call JSON"
    if any(call.get("name") != "search" for call in tool_calls):
        return False, "Unexpected tool name"
    if len(tool_calls) != len(_tool_response_contents(content)):
        return False, "tool_call/tool_response count mismatch"
    if extract_solution(text) is None:
        return False, "Missing final answer"
    return True, "Valid tool-call sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    # Try <answer>...</answer> first
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if matches:
        return matches[-1].group(1).strip()
    # If not found, try \\boxed{...}
    boxed_pattern = r"\\boxed\{([^}]*)\}"
    boxed_matches = re.findall(boxed_pattern, solution_str)
    if boxed_matches:
        return boxed_matches[-1].strip()
    return None


def extract_information_blocks(text: str) -> list[str]:
    information = re.findall(r"<information>(.*?)</information>", text, re.DOTALL)
    tool_responses = _tool_response_contents(text)
    return [match.strip() for match in information + tool_responses]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(
    solution_str,
    ground_truth,
    method="strict",
    structure_format_score=0,
    final_format_score=0,
    retrieval_score=0,
    format_score=0,
    score=1.0,
):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth["target"])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score  # 0.3
            else:
                return structure_format_score  # 0.2
        else:
            return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if is_valid_format:
                return score  # 1
            else:
                return score - structure_format_score  # 0.8
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score  # 0.3
            else:
                return structure_format_score  # 0.2
        else:
            return final_format_score  # 0.1
