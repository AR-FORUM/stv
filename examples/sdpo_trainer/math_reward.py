# Math reward function for SDPO training using math-verify library.
# Compatible with verl-private's math data format.
#
# Install: pip install math-verify
#
# Expected data format (parquet):
#   - data_source: str (e.g., "math_dapo", "dapo17k_en", etc.)
#   - prompt: list[dict] with role/content (chat format)
#   - reward_model.ground_truth: str (the final answer without \boxed{})
#   - extra_info: dict (optional metadata)

from typing import Optional

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    raise ImportError(
        "math-verify is required for this reward function. "
        "Install it with: pip install math-verify"
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Compute the reward score for a math solution using math-verify.

    Args:
        data_source: Dataset identifier (not used, for API compatibility)
        solution_str: The model's solution/response string
        ground_truth: The correct answer (without \\boxed{})
        extra_info: Additional metadata (not used)
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        float: 1.0 if correct, 0.0 if incorrect
    """
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [solution_str])
    except TimeoutException:
        ret_score = 0.0
    except Exception:
        ret_score = 0.0

    return float(ret_score)
