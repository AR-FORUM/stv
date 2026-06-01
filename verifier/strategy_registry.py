"""Strategy registry for the verifier.

Only the two strategies used by the method are kept:

- ``O``: the unconditioned verifier (sees problem + solution). This is the STV verifier
  used at both training and evaluation time.
- ``Y6``: the privileged teacher (also sees the confirmed verdict and the full reference
  solution), used to build the distillation target during STV training.
"""

from __future__ import annotations

STRATEGIES = {
    "O": {
        "name": "Unconditioned (last turn)",
        "feedback_type": "identify_mistake",
        "prompt_id": "identify_mistake",
        "reprompt_mode": "last_turn",
    },
    "Y6": {
        "name": "O-style + Full Sol. (privileged)",
        "feedback_type": "noleak_verdict_full_ostyle",
        "prompt_id": "noleak_verdict_full_ostyle",
        "reprompt_mode": "last_turn",
        "mask_keep_fraction": 1.0,
    },
}
