"""
Analyzer Protocol — the interface every analysis engine must implement.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from .schemas import AnalysisResult, DetectionRequest


@runtime_checkable
class Analyzer(Protocol):
    """
    Protocol that both StructuredAnalyzer and SemanticAnalyzer implement.

    Using ``Protocol`` instead of an ABC so that existing classes can
    conform without inheriting from a shared base.
    """

    def analyze(self, request: DetectionRequest) -> AnalysisResult:
        """Run analysis on a single detection request."""
        ...

    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready for inference."""
        ...

    def health(self) -> Dict[str, Any]:
        """Return a health-check dict for the ``/health`` endpoint."""
        ...
