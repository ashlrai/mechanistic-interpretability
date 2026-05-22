import pytest

from mech_interp.analysis import logit_diff_recovery


def test_logit_diff_recovery_computes_fraction() -> None:
    result = logit_diff_recovery(
        clean_logits=[0.0, 5.0, 1.0],
        corrupted_logits=[0.0, 2.0, 3.0],
        patched_logits=[0.0, 4.0, 2.0],
        correct_token_index=1,
        incorrect_token_index=2,
    )

    assert result.clean_logit_diff == 4.0
    assert result.corrupted_logit_diff == -1.0
    assert result.patched_logit_diff == 2.0
    assert result.recovery_fraction == 0.6


def test_logit_diff_recovery_handles_zero_denominator() -> None:
    result = logit_diff_recovery(
        clean_logits=[1.0, 1.0],
        corrupted_logits=[1.0, 1.0],
        patched_logits=[2.0, 1.0],
        correct_token_index=0,
        incorrect_token_index=1,
    )

    assert result.recovery_fraction == 0.0


def test_logit_diff_recovery_rejects_missing_token_index() -> None:
    with pytest.raises(ValueError, match="outside"):
        logit_diff_recovery(
            clean_logits=[1.0],
            corrupted_logits=[1.0],
            patched_logits=[1.0],
            correct_token_index=0,
            incorrect_token_index=2,
        )
