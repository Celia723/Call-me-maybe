
import numpy as np


def mask_logits(
    logits: list[float],
    allowed_token_ids: list[int]
) -> np.ndarray:
    """Allow only the specified token IDs."""

    scores = np.asarray(logits, dtype=float)
    masked_scores = np.full(scores.shape, -np.inf)

    valid_ids = sorted({
        token_id
        for token_id in allowed_token_ids
        if isinstance(token_id, int)
        and 0 <= token_id < len(scores)
    })

    if not valid_ids:
        raise ValueError("No valid tokens are allowed.")

    masked_scores[valid_ids] = scores[valid_ids]

    return masked_scores


def get_allowed_next_tokens(
    candidates: list[list[int]],
    generated_ids: list[int]
) -> list[int]:
    """Find tokens that continue at least one valid candidate."""

    allowed_ids = set()

    for candidate in candidates:
        if candidate[:len(generated_ids)] == generated_ids:
            if len(candidate) > len(generated_ids):
                allowed_ids.add(candidate[len(generated_ids)])

    return sorted(allowed_ids)
