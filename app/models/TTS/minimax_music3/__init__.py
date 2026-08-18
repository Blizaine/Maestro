"""Native and accelerated MiniMax-Music3 components used by Maestro.

The neural-network definitions are adapted from the Apache-2.0 Diffusers
integration contributed by the MiniMax and Hugging Face teams. Model weights
remain governed by MiniMax's Music3 Community License. Maestro's default
pipeline uses the ConvRot checkpoints and semantic decoder adapted from
WanGP; the original Diffusers-style pipeline remains available as a fallback.
"""

from .condition_encoder import MiniMaxMusic3ConditionEncoder
from .optimized_pipeline import MiniMaxMusic3Pipeline
from .pipeline import MiniMaxMusic3Pipeline as LegacyMiniMaxMusic3Pipeline
from .rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
from .transformer import MiniMaxMusic3Transformer1DModel
from .vocoder import MiniMaxMusic3Vocoder

__all__ = [
    "MiniMaxMusic3ConditionEncoder",
    "LegacyMiniMaxMusic3Pipeline",
    "MiniMaxMusic3Pipeline",
    "MiniMaxMusic3RVQDepthDecoder",
    "MiniMaxMusic3Transformer1DModel",
    "MiniMaxMusic3Vocoder",
]
