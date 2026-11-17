"""RAGAS import patch - must be imported before RAGAS.

RAGAS 0.2.x has an outdated import for VertexAI that fails on modern langchain.
Patch the module namespace before RAGAS loads to avoid ModuleNotFoundError.
"""

import importlib.util
import sys

_spec = importlib.util.spec_from_loader("langchain_community.chat_models.vertexai", loader=None)
if _spec:
    _mod = type(sys)("langchain_community.chat_models.vertexai")
    _mod.ChatVertexAI = None  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _mod
