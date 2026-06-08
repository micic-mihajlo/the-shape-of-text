"""Distribution style-alignment components for causal language models."""

from importlib import import_module

_EXPORT_MODULES = {
    "AlignmentLossOutput": "shape_of_text.losses",
    "CausalLMAlignmentLoss": "shape_of_text.losses",
    "JMQLoss": "shape_of_text.losses",
    "MMDLoss": "shape_of_text.losses",
    "aggregate_style_metrics": "shape_of_text.style_metrics",
    "evaluate_kl": "shape_of_text.evaluation",
    "kl_from_logits": "shape_of_text.evaluation",
    "post_style_metrics": "shape_of_text.style_metrics",
    "style_delta": "shape_of_text.style_metrics",
}

__all__ = [
    "AlignmentLossOutput",
    "CausalLMAlignmentLoss",
    "JMQLoss",
    "MMDLoss",
    "aggregate_style_metrics",
    "evaluate_kl",
    "kl_from_logits",
    "post_style_metrics",
    "style_delta",
]


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    return getattr(module, name)
