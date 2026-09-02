"""Typed data for the three verify/ feedback builders:

  - CompilationResult.get_error_summary   (synthesis/verify/compiler.py)
  - VerificationResult.get_error_summary  (synthesis/verify/verifier.py)
  - VerificationResult.get_structured_feedback (synthesis/verify/verifier.py)

As with `models/feedback.py`, every field here is a value the surrounding
method already computes; this module only carries that data to the
`verify/*.j2` templates. The short-circuit branches (success, raw-output-only,
no-errors fallback, and the "" early return in get_structured_feedback) stay
plain Python returns in the calling method — only the multi-line "\\n".join(...)
branches are routed through a template, matching the exact idiom used by
`EvaluationResult.get_feedback_summary`.
"""
from typing import List, Optional

from synthesis.prompt_rendering.base import PromptModel


class ErrorEntry(PromptModel):
    """One `  - Line {line}: {message}` row, shared by both error-summary builders."""

    line: int
    message: str


class CompilationErrorSummaryModel(PromptModel):
    """Data for CompilationResult.get_error_summary's multi-error branch."""

    error_count: int
    errors: List[ErrorEntry]


class VerificationErrorSummaryModel(PromptModel):
    """Data for VerificationResult.get_error_summary's multi-error branch."""

    error_count: int
    errors: List[ErrorEntry]


class DiagnosticEntry(PromptModel):
    """One numbered diagnostic block in the structured verification analysis.

    Every optional field is pre-decided in Python (truthy value or None/empty
    list) so the template only has to test presence, matching the original
    method's `if diagnostic.<field>:` branches exactly.
    """

    index: int
    location: str
    obligation_kind_title: str
    message: str
    call_name: Optional[str]
    failing_text: Optional[str]
    related_display: Optional[str]
    source_excerpt_lines: List[str]
    contract_excerpt_lines: List[str]
    remediation: Optional[str]


class StructuredFeedbackModel(PromptModel):
    """Data for VerificationResult.get_structured_feedback's non-empty branch."""

    diagnostics: List[DiagnosticEntry]
