"""Runtime compatibility checks for the pinned H3 Singularity export."""


def validate_minimax_h3_singularity_checkpoint(
    checkpoint: dict,
    qkv_layout: str,
) -> None:
    """Reject files that do not match the pinned pruned INT8 ConvRot export."""

    if str(qkv_layout or "").strip().lower() != "grouped":
        raise ValueError(
            "MiniMax H3 Singularity requires logical grouped [Q, K, V] rows."
        )
    if checkpoint.get("convrot") is not True:
        raise ValueError(
            "MiniMax H3 Singularity requires its INT8 ConvRot checkpoint; "
            "stock and fused H3 checkpoints are not compatible fallbacks."
        )
    formats = {
        item.strip().lower()
        for item in str(checkpoint.get("quantization_format") or "").split(",")
        if item.strip()
    }
    if "int8_tensorwise" not in formats:
        raise ValueError(
            "MiniMax H3 Singularity requires int8_tensorwise ConvRot weights."
        )
    if checkpoint.get("convrot_group_size") != 256:
        raise ValueError(
            "MiniMax H3 Singularity requires ConvRot group size 256."
        )
    if (
        checkpoint.get("compressed_modulation") is not True
        or checkpoint.get("adaln_curve_grid") != 1025
        or checkpoint.get("time_embed_dim") != 8
    ):
        raise ValueError(
            "MiniMax H3 Singularity requires the pruned [1025, 8] AdaLN curve table."
        )


__all__ = [
    "validate_minimax_h3_singularity_checkpoint",
]
