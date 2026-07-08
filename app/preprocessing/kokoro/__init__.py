# NOTE: This is the Scenema-aligned Kokoro TTS copy. It forces CPU placement
# so mmgp_offload can control device transfers from outside the model.
# Maestro also has a SECOND copy at `app/models/wan/multitalk/kokoro/` that is
# intentionally divergent — it auto-picks CUDA and is used by multitalk's
# in-process TTS path. Do NOT cross-wire them: changing this copy to auto-CUDA
# breaks Scenema's offload contract; changing multitalk's copy to force-CPU
# breaks multitalk's existing latency budget.

__version__ = '0.9.4'

from loguru import logger
import sys

# Remove default handler
logger.remove()

# Add custom handler with clean format including module and line number
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <cyan>{module:>16}:{line}</cyan> | <level>{level: >8}</level> | <level>{message}</level>",
    colorize=True,
    level="INFO" # "DEBUG" to enable logger.debug("message") and up prints 
                 # "ERROR" to enable only logger.error("message") prints
                 # etc
)

# Disable before release or as needed
logger.disable("kokoro")

from .model import KModel
from .pipeline import KPipeline
