"""MOSS enhancer canvas context and model-facing prompt compilation."""
import json
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
        "For a NEW layout, return a draft with this exact width and height, "
        'bbox_coord_space="abs" and bbox_format="xyxy". Draft boxes are '
        "[x_min,y_min,x_max,y_max] from the top left, "
        f"with X in 0..{width} and Y in 0..{height}. Maestro performs the coordinate conversion. "
        "Do not manually center or reorder draft coordinates. "
        "An EXISTING compiled layout keeps bbox_format=\"yxyx\" "
        "and bbox_coord_space=\"abs_centralized\", with [y_min,x_min,y_max,x_max] "
        "integer pixel corners measured from the canvas center. "
        f"X bounds are {-width//2} to {width//2}; Y bounds are {-height//2} to {height//2}. "
        "Keep every box inside those bounds; preserve exact supplied text and extra layout fields. "
        "A plain caption request stays a natural-language caption. Do not turn an ordinary "
        "photo into a layout; use a layout when requested or when explicit placement needs boxes."
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


def finalize_enhancement_output(output, resolution=None):
    """Compile new drafts, retaining the native syntax of saved editor layouts.

    This is an enhancer boundary, not a parser for all generation prompts. The
    latter also accept prose combined with JSON. Never run the generic prose
    cleaner over this output: # comments and ** inside display strings are data.
    """
    from services.moss_image_layout import compile_layout_draft, strict_loads

    text = str(output or '').strip()
    fence = re.fullmatch(r'```(?:json|text)?\s*\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    if not text:
        raise ValueError('MOSS Image enhancer returned an empty prompt.')
    lines = text.splitlines()
    comments = [line for line in lines if line.lstrip().startswith('#')]
    candidate = '\n'.join(line for line in lines if not line.lstrip().startswith('#')).strip()
    if not candidate:
        raise ValueError('MOSS Image enhancer returned comments without a prompt.')
    if not candidate.startswith(('{', '[')):
        return text

    document = strict_loads(candidate)
    if not isinstance(document, dict):
        raise ValueError('MOSS Image layout must be a JSON object.')
    if set(document) == {'caption'}:
        caption = document['caption']
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError('MOSS Image caption must be nonempty text.')
        compiled = caption
    elif (document.get('bbox_coord_space') == 'abs' or document.get('bbox_format') == 'xyxy'
          or ('width' in document or 'height' in document)
          and document.get('bbox_coord_space') != 'abs_centralized'):
        compiled = compile_layout_draft(document, resolution or '1024x1024')
    else:
        # The visual helper and older saved prompts use final centered boxes.
        # Preserve their extra fields, literal strings and coordinates rather
        # than feeding them through the stricter new-draft vocabulary.
        width, height = _canvas_size(resolution)
        if not isinstance(document.get('summary'), dict) or not isinstance(document.get('items'), list):
            raise ValueError('MOSS Image layout requires summary and a flat items array.')
        if document.get('bbox_coord_space', 'abs_centralized') != 'abs_centralized' or document.get('bbox_format', 'yxyx') != 'yxyx':
            raise ValueError('MOSS Image compiled layouts use abs_centralized/yxyx coordinates.')
        for index, item in enumerate(document['items']):
            if not isinstance(item, dict):
                raise ValueError(f'MOSS Image item {index + 1} must be an object.')
            box = item.get('bbox')
            if not isinstance(box, list) or len(box) != 4 or any(type(edge) is not int for edge in box):
                raise ValueError(f'MOSS Image item {index + 1} needs four integer centered pixel corners.')
            top, left, bottom, right = box
            if not (-height // 2 <= top < bottom <= height // 2 and -width // 2 <= left < right <= width // 2):
                raise ValueError(f'MOSS Image item {index + 1} exceeds the selected {width}x{height} canvas or has no area.')
        compiled = json.dumps(document, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return '\n'.join([*comments, compiled])
