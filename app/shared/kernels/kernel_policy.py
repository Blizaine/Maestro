"""Policy for optional kernel optimizations and their numerical trade-offs."""

CHOICES = [
    ("Preserve Precision", "strict"),
    ("Allow Faster Approximate Kernels", "fast"),
]

precision = "fast"
RANKING = ("wgpkernels", "kitchen", "triton", "pytorch")
families = RANKING


def configure_families(selection: str) -> None:
    """Put the explicitly selected INT8 family first for other fused kernels."""
    global families
    if selection in RANKING[:3]:
        families = (selection,) + tuple(family for family in RANKING if family != selection)
    else:
        families = RANKING


def first(candidates: dict[str, bool]) -> str | None:
    """Return the preferred available family, or None to use the PyTorch path."""
    return next((family for family in families if candidates.get(family)), None)


def configure(value: str) -> None:
    """Select whether approximate non-INT8 kernel paths may run."""

    global precision
    if value not in {"strict", "fast"}:
        raise ValueError(f"Unknown non-INT8 kernel precision: {value!r}")
    precision = value


def allow_approximate() -> bool:
    return precision == "fast"


__all__ = [
    "CHOICES",
    "RANKING",
    "allow_approximate",
    "configure",
    "configure_families",
    "families",
    "first",
]
