"""Request-to-loader handoff for opt-in H3 runtime components."""
from collections.abc import Mapping


def two_pass_load_context(model_def, custom_settings, loaded_model=None):
    """Return explicit loader options and whether the loaded pipe must change.

    The flag is part of loaded-model identity, because an ordinary H3 instance
    must never gain a latent upscaler during an active render. Missing settings
    mean Off and do not authorize an optional asset download.
    """
    if not str(model_def.get('architecture') or '').startswith('minimax_h3'):
        return {}, False
    from models.minimax_h3.two_pass import parse_two_pass_toggle

    settings = custom_settings if isinstance(custom_settings, Mapping) else {}
    enabled = parse_two_pass_toggle(settings.get('h3_two_pass_enabled'))
    if enabled and model_def.get('minimax_h3_two_pass_compatible') is not True:
        raise ValueError('This H3 recipe does not support Two-Pass refinement.')
    reload_needed = (
        loaded_model is not None
        and getattr(loaded_model, 'two_pass_enabled', False) is not enabled
    )
    return {'minimax_h3_two_pass': enabled}, reload_needed
