"""Process-wide policy for optional kernels that can change rounding slightly."""

CHOICES = [
    ("Preserve Precision", "strict"),
    ("Allow Faster Approximate Kernels", "fast"),
]

precision = "fast"


def configure(value: str) -> None:
    """Select whether approximate non-INT8 kernel paths may run."""

    global precision
    if value not in {"strict", "fast"}:
        raise ValueError(f"Unknown non-INT8 kernel precision: {value!r}")
    precision = value


def allow_approximate() -> bool:
    return precision == "fast"


__all__ = ["CHOICES", "allow_approximate", "configure"]
