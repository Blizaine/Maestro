"""Small, low-risk runtime switches for LTX-2 masked editing.

These defaults are scoped to Maestro's LTX-2.3 mask-preserving Outpaint path.
The production path follows Lightricks' published two-stage graph: the exact
green missing-region marker, full reference attention, a decoded-pixel
Laplacian/Lanczos handoff, and final Laplacian source restoration.
"""

LTX2_OUTPAINTING_LAPLACIAN_BLEND = True
LTX2_OUTPAINTING_LAPLACIAN_MASK_LOW_RES_DILATION = 0
LTX2_OUTPAINTING_SOURCE_FEATHER_PIXELS = 8
# Keep generated Outpaint colors model-native. Marker cleanup remains
# available for diagnostics, but the production path must be evaluated
# without semantic color correction.
LTX2_OUTPAINTING_MARKER_RESIDUE_CLEANUP = False
# Retained only for the legacy rollback path. Official Outpaint passes
# match_generated_canvas=False and never grades generated pixels.
LTX2_OUTPAINTING_CANVAS_MATCH = True
LTX2_OUTPAINTING_CANVAS_MATCH_MAX_GAIN = 1.15
LTX2_OUTPAINTING_CANVAS_MATCH_MAX_OFFSET = 0.08
LTX2_OUTPAINTING_CANVAS_MATCH_MAX_CHROMA_SHIFT = 0.12
LTX2_OUTPAINTING_CANVAS_MATCH_MAX_GRADIENT_CHROMA_SHIFT = 0.18
LTX2_OUTPAINTING_CANVAS_MATCH_MAX_SHARPEN = 0.60
LTX2_LAPLACIAN_BLEND_MASK_LOW_RES_LONG_SIDE = 64
LTX2_MASKED_CONTROL_VIDEO_PAD_RGB = (128, 128, 128)
# Exact marker used by Lightricks' LTXVInpaintPreprocess node.
LTX2_INPAINT_CONTROL_VIDEO_PAD_RGB = (102, 255, 0)
