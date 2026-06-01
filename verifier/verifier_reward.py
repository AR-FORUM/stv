# Verifier reward function for verl training.
#
# Parses the model's VERDICT from its response and compares against the
# ground truth label ("CORRECT" or "INCORRECT").
# Also enforces an explicit FEEDBACK field in the output format.
#
# Expected data format (parquet):
#   - data_source: "verifier"
#   - prompt: list[dict] (verification prompt in chat format)
#   - reward_model.ground_truth: "CORRECT" or "INCORRECT"
#   - extra_info: dict (optional metadata)

import re
from typing import Any, Optional


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Compute reward for a verification response.

    The model is expected to output:
        VERDICT: CORRECT or INCORRECT
        FEEDBACK: ...

    Returns binary reward:
      - 1.0 only if verdict matches ground truth AND FEEDBACK field is present
      - 0.0 otherwise
    """
    del data_source, extra_info, kwargs
    return compute_base_reward(solution_str, ground_truth)["score"]


def compute_base_reward(solution_str: str, ground_truth: str) -> dict[str, Any]:
    """Evaluate the current binary verifier reward components."""
    predicted = parse_verdict(solution_str)
    verdict_correct = (
        predicted is not None and predicted == ground_truth.strip().upper()
    )
    feedback_text = extract_feedback_text(solution_str)
    has_feedback = feedback_text is not None
    return {
        "score": 1.0 if verdict_correct and has_feedback else 0.0,
        "predicted_verdict": predicted,
        "verdict_correct": verdict_correct,
        "has_feedback": has_feedback,
        "feedback_text": feedback_text,
    }


def parse_verdict(text: str) -> Optional[str]:
    """Extract VERDICT from model output."""
    for line in text.split("\n"):
        ls = line.strip()
        if ls.upper().startswith("VERDICT:"):
            vt = ls[len("VERDICT:"):].strip().upper()
            if "INCORRECT" in vt:
                return "INCORRECT"
            elif "CORRECT" in vt:
                return "CORRECT"
            break
    # Fallback: check for keywords anywhere
    lower = text.lower()
    if "incorrect" in lower:
        return "INCORRECT"
    if "correct" in lower:
        return "CORRECT"
    return None


def extract_feedback_text(text: str) -> Optional[str]:
    """Extract FEEDBACK text using the current training-time field rule."""
    match = re.search(r"(?is)\bFEEDBACK:\s*(.*)", text)
    if not match:
        return None
    feedback = match.group(1).strip()
    return feedback or None
