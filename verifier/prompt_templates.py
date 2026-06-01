"""Oracle prompt templates for verifier training and evaluation.

Two prompts are used by the method:

- ``identify_mistake``: the unconditional verifier prompt. The verifier sees only the
  problem and the student's solution, and must produce a VERDICT and FEEDBACK.
- ``noleak_verdict_full_ostyle``: the reference-conditioned teacher prompt. In addition
  to the problem and solution, the teacher also sees the confirmed verdict and a
  reference solution. It is used to build the privileged teacher target for STV
  distillation, while being instructed not to leak the reference.
"""

ORACLE_IDENTIFY_MISTAKE = """\
You are reviewing a math student's solution.

Problem:
{problem}

Student's solution:
{solution}

Tasks:
1. Determine if the final answer is correct.
2. If incorrect, identify the specific mistake(s) in reasoning or calculation.

Rules:
- Do NOT reveal the correct answer or solve the problem yourself.
- Keep feedback to 2-3 sentences maximum.

Format your response exactly as:
VERDICT: CORRECT or INCORRECT
FEEDBACK: [If incorrect, describe the mistake. If correct, write "None".]"""

ORACLE_NOLEAK_VERDICT_FULL_OSTYLE = """\
You are reviewing a math student's solution.

Problem:
{problem}

Student's solution:
{solution}

Verification status: The student's final answer has been independently confirmed \
as {verdict}.

Reference solution:
{oracle_solution}

Tasks:
1. Determine if the final answer is correct.
2. If incorrect, identify the specific mistake(s) in reasoning or calculation.

Rules:
- Use the reference solution only to help judge the student's work.
- Write feedback as a direct critique of the student's reasoning, not as a \
comparison to the reference solution.
- Do NOT mention the reference solution explicitly.
- Do NOT reveal the correct answer or solve the problem yourself.
- Keep feedback to 2-3 sentences maximum.

Format your response exactly as:
VERDICT: CORRECT or INCORRECT
FEEDBACK: [If incorrect, describe the mistake. If correct, write "None".]"""

ORACLE_TEMPLATES = {
    "identify_mistake": ORACLE_IDENTIFY_MISTAKE,
    "noleak_verdict_full_ostyle": ORACLE_NOLEAK_VERDICT_FULL_OSTYLE,
}

VERIFY_UNCONDITIONAL_PROMPT_ID = "identify_mistake"


def get_oracle_template(prompt_id: str) -> str:
    """Return a registered oracle prompt template by id."""
    if prompt_id not in ORACLE_TEMPLATES:
        raise KeyError(
            f"Unknown prompt_id '{prompt_id}'. Available: {sorted(ORACLE_TEMPLATES.keys())}"
        )
    return ORACLE_TEMPLATES[prompt_id]


def render_oracle_prompt(prompt_id: str, **kwargs) -> str:
    """Render a prompt template by id with explicit kwargs."""
    return get_oracle_template(prompt_id).format(**kwargs)
