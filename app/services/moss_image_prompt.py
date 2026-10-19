"""Canvas context for MOSS's centered-pixel layout prompting dialect."""
import re


def _canvas_size(resolution=None):
    match = re.fullmatch(r"(\d+)\s*[xX×]\s*(\d+)", str(resolution or "1024x1024"))
    if not match:
        raise ValueError("MOSS Image canvas resolution must be WIDTHxHEIGHT.")
    width, height = map(int, match.groups())
    if width < 16 or height < 16 or width % 16 or height % 16 or width * height > 2048**2:
        raise ValueError("MOSS Image requires canvas dimensions divisible by 16, up to about 4 megapixels.")
    return width, height


def enhancement_canvas_context(resolution=None):
    width, height = _canvas_size(resolution)
    return (
        f"Selected MOSS Image canvas: {width}x{height} pixels. "
        "For an existing or explicitly requested layout, use bbox_format=\"yxyx\" "
        "and bbox_coord_space=\"abs_centralized\". Boxes are [y_min,x_min,y_max,x_max] "
        "integer pixel corners measured from the canvas center. "
        f"X bounds are {-width//2} to {width//2}; Y bounds are {-height//2} to {height//2}. "
        "Keep every box inside those bounds; preserve exact supplied text and extra layout fields. "
        "A plain caption request stays a natural-language caption."
    )


def render_layout_instructions(instructions, resolution=None):
    """Resolve the upstream classic enhancer's canvas placeholders locally."""
    context = enhancement_canvas_context(resolution)
    width, height = _canvas_size(resolution)
    values = {'width': width, 'height': height, 'half_width': width // 2,
              'half_height': height // 2, 'aspect_ratio': f'{width}:{height}'}
    for name, value in values.items():
        instructions = str(instructions or '').replace('{{' + name + '}}', str(value))
    return f'{instructions}\n\n{context}'
