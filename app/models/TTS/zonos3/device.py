"""Resolve the execution device for MMGP-offloaded ZONOS3 modules."""

import torch


def execution_device(module, fallback=None):
    """Use MMGP's active execution target before inspecting resident parameters.

    Offloaded modules keep parameters on the host between calls while MMGP's
    hooks stage their weights on the execution device for each forward.
    """
    hook = getattr(module, "_hf_hook", None)
    hook_device = getattr(hook, "execution_device", None)
    if hook_device is not None:
        return torch.device(hook_device)

    explicit_device = getattr(module, "execution_device", None)
    if explicit_device is not None:
        return torch.device(explicit_device)

    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    if fallback is not None:
        return torch.device(fallback)
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
