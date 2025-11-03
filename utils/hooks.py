"""Helper functions for attaching forward hooks and custom adapters.

This module contains utilities to register PyTorch forward hooks on modules in
order to capture activations or intervene with custom logic.  It is useful
when inserting parameter‑efficient adapters, collecting intermediate values
or measuring communication/activation sizes.

Usage:

    from utils import hooks

    activations = {}

    def save_output(module, inp, out):
        activations[module] = out.detach()

    model = ...  # any nn.Module
    handle = hooks.register_forward_hook(model.layers[0], save_output)
    ...  # run model forward
    handle.remove()

"""

from typing import Callable, Dict, Any
import torch.nn as nn


def register_forward_hook(module: nn.Module, hook_fn: Callable) -> Any:
    """Register a forward hook on a module and return the handle.

    Args:
        module: The nn.Module on which to register the hook.
        hook_fn: A callable with signature (module, input, output) -> None.

    Returns:
        A handle that can be used to remove the hook.
    """
    return module.register_forward_hook(hook_fn)


def collect_module_outputs(model: nn.Module, module_names: Dict[str, str]) -> Dict[str, Any]:
    """Attach hooks to named submodules and collect their outputs during forward.

    Args:
        model: The parent model.
        module_names: A mapping from a friendly name to the attribute path of the
            submodule (e.g. {"q_proj": "model.layers.0.self_attn.q_proj"}).

    Returns:
        A dictionary mapping friendly names to collected outputs.  Hooks are
        automatically removed after the forward pass.
    """
    outputs: Dict[str, Any] = {}
    handles = []

    def make_hook(name: str):
        def hook_fn(module, inp, out):
            outputs[name] = out.detach()
        return hook_fn

    # Attach hooks
    for friendly_name, attr_path in module_names.items():
        submodule = model
        for attr in attr_path.split('.'):
            submodule = getattr(submodule, attr)
        handle = submodule.register_forward_hook(make_hook(friendly_name))
        handles.append(handle)

    # Run a dummy forward call to capture outputs
    def runner(*args, **kwargs):
        nonlocal handles
        try:
            return model(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()
            handles = []

    return outputs, runner
