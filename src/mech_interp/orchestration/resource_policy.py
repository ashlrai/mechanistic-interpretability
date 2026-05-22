from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_DTYPE = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


@dataclass(frozen=True)
class ActivationEstimate:
    batch_size: int
    sequence_length: int
    hidden_size: int
    hook_count: int
    dtype: str = "float16"

    @property
    def bytes(self) -> int:
        if self.dtype not in BYTES_PER_DTYPE:
            supported = ", ".join(sorted(BYTES_PER_DTYPE))
            raise ValueError(f"Unsupported dtype '{self.dtype}'. Supported: {supported}.")
        return (
            self.batch_size
            * self.sequence_length
            * self.hidden_size
            * self.hook_count
            * BYTES_PER_DTYPE[self.dtype]
        )

    @property
    def gib(self) -> float:
        return self.bytes / 1024**3


@dataclass(frozen=True)
class ResourcePolicy:
    max_ram_gib: float = 128.0
    max_activation_fraction: float = 0.35

    @property
    def max_activation_gib(self) -> float:
        return self.max_ram_gib * self.max_activation_fraction

    def validate_activation_estimate(self, estimate: ActivationEstimate) -> None:
        if estimate.gib > self.max_activation_gib:
            raise ValueError(
                "Estimated activation cache exceeds policy: "
                f"{estimate.gib:.2f} GiB > {self.max_activation_gib:.2f} GiB."
            )
