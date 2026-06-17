from typing import TYPE_CHECKING
from transformers.utils import _LazyModule, is_torch_available

_import_structure = {
    "configuration_chimera": ["ChimeraConfig"],
}

if is_torch_available():
    _import_structure["modeling_chimera"] = [
        "ChimeraModel",
        "ChimeraForCausalLM",
        "ChimeraPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_chimera import ChimeraConfig
    if is_torch_available():
        from .modeling_chimera import ChimeraForCausalLM, ChimeraModel, ChimeraPreTrainedModel
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
