from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LogitDiffResult:
    clean_logit_diff: float
    corrupted_logit_diff: float
    patched_logit_diff: float
    recovery_fraction: float


def logit_diff_recovery(
    clean_logits: list[float],
    corrupted_logits: list[float],
    patched_logits: list[float],
    correct_token_index: int,
    incorrect_token_index: int,
) -> LogitDiffResult:
    clean = _logit_diff(clean_logits, correct_token_index, incorrect_token_index)
    corrupted = _logit_diff(corrupted_logits, correct_token_index, incorrect_token_index)
    patched = _logit_diff(patched_logits, correct_token_index, incorrect_token_index)
    denominator = clean - corrupted
    recovery = 0.0 if denominator == 0 else (patched - corrupted) / denominator
    return LogitDiffResult(
        clean_logit_diff=clean,
        corrupted_logit_diff=corrupted,
        patched_logit_diff=patched,
        recovery_fraction=recovery,
    )


def _logit_diff(logits: list[float], correct_token_index: int, incorrect_token_index: int) -> float:
    try:
        return float(logits[correct_token_index] - logits[incorrect_token_index])
    except IndexError as exc:
        raise ValueError("Token index is outside the logits vector.") from exc
