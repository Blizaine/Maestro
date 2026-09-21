"""Small Diffusers 0.36 compatibility bridge; never patches other pipelines."""

from functools import wraps

from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers


def apply_lora_scale(argument_name):
    # MMGP normally owns Maestro's adapters. Retain upstream PEFT semantics
    # for direct pipeline callers without requiring a global Diffusers update.
    def decorate(function):
        @wraps(function)
        def forward(self, *args, **kwargs):
            options = dict(kwargs.get(argument_name) or {})
            scale = options.pop("scale", 1.0)
            kwargs[argument_name] = options
            if USE_PEFT_BACKEND:
                scale_lora_layers(self, scale)
            try:
                return function(self, *args, **kwargs)
            finally:
                if USE_PEFT_BACKEND:
                    unscale_lora_layers(self, scale)
        return forward
    return decorate
