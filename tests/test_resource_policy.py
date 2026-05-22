import pytest

from mech_interp.orchestration import ActivationEstimate, ResourcePolicy


def test_activation_estimate_computes_gib() -> None:
    estimate = ActivationEstimate(
        batch_size=2,
        sequence_length=10,
        hidden_size=100,
        hook_count=3,
        dtype="float16",
    )

    assert estimate.bytes == 12_000
    assert estimate.gib > 0


def test_resource_policy_rejects_large_estimates() -> None:
    estimate = ActivationEstimate(
        batch_size=1024,
        sequence_length=8192,
        hidden_size=8192,
        hook_count=80,
        dtype="float16",
    )

    with pytest.raises(ValueError, match="exceeds policy"):
        ResourcePolicy(max_ram_gib=128).validate_activation_estimate(estimate)
