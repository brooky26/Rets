"""
Optional-torch import shim.

PyTorch is a heavy dependency the platform does not want to force onto
every deployment (e.g. a minimal paper-trading Railway instance that
never trains sequence models). Every other module in `sequence_models/`
imports `torch` from here rather than directly, so the *symptom* of a
missing torch install is always the same clear `SequenceModelsUnavailableError`
raised at model-instantiation time — not a bare `ModuleNotFoundError`
traceback from deep inside an unrelated import chain.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in torch-less environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


class SequenceModelsUnavailableError(RuntimeError):
    pass


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise SequenceModelsUnavailableError(
            "PyTorch is not installed, so sequence models (LSTM/Transformer) are unavailable. "
            "Install it with `pip install torch` to enable this optional model family — the rest "
            "of the platform (Bayesian Logistic, Bagged GBM, regime detection, risk, execution, "
            "etc.) has no dependency on torch and is unaffected."
        )
