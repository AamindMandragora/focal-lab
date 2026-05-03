"""
Verification wrapper for Python-first CSD synthesis.

Transpiles generated Python strategy code to Dafny, runs `dafny verify`,
and parses the results.
"""

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from verification.dafny_runner import (
    check_dafny_available,
    get_verified_agent_synthesis_dafny_path,
    prepare_temp_dafny_dir,
    prepare_temp_dafny_dir_from_dafny,
)


@dataclass
class VerificationDiagnostic:
    """Structured summary of one verifier finding for refinement prompts."""

    file: str
    line: int
    column: int
    message: str
    obligation_kind: str
    failing_text: str = ""
    source_excerpt: str = ""
    call_name: str = ""
    related_file: Optional[str] = None
    related_line: Optional[int] = None
    related_message: str = ""
    contract_excerpt: str = ""
    python_line: Optional[int] = None


@dataclass
class VerificationError:
    """A single verification error from Dafny."""
    file: str
    line: int
    column: int
    message: str
    error_type: str = "Error"
    python_line: Optional[int] = None
    
    def __str__(self) -> str:
        return f"{self.file}({self.line},{self.column}): {self.error_type}: {self.message}"


@dataclass
class VerificationResult:
    """Result of running Dafny verification."""
    success: bool
    errors: list[VerificationError] = field(default_factory=list)
    raw_output: str = ""
    raw_stderr: str = ""
    return_code: int = 0
    diagnostics: list[VerificationDiagnostic] = field(default_factory=list)
    
    def get_error_summary(self) -> str:
        """Get a human-readable summary of errors for the LLM refinement prompt."""
        if self.success:
            return "Verification successful"
        
        combined_raw = (self.raw_output or "") + "\n" + (self.raw_stderr or "")
        combined_raw = combined_raw.strip()

        if not self.errors:
            # No parsed errors, return raw output so the model still sees Dafny's message
            return combined_raw or "Verification failed (no details captured)."

        lines = [f"Dafny verification failed with {len(self.errors)} error(s):", ""]
        for err in self.errors:
            location = f"Dafny line {err.line}, Column {err.column}"
            if err.python_line is not None:
                location += f"; Python line {err.python_line}"
            lines.append(f"  ({location}): {err.error_type}: {err.message}")
        lines.append("")
        # Append raw Dafny output so the model sees the exact errors (e.g. multi-line, related hints)
        if combined_raw:
            raw_preview = combined_raw if len(combined_raw) <= 3500 else combined_raw[:3500] + "\n... (truncated)"
            lines.append("Full Dafny output:")
            lines.append(raw_preview)
        return "\n".join(lines)

    def get_structured_feedback(self) -> str:
        """Return a compact structured verifier summary for refinement prompts."""
        if self.success or not self.diagnostics:
            return ""

        lines = ["Structured verification analysis:"]
        for idx, diagnostic in enumerate(self.diagnostics, start=1):
            location = f"{Path(diagnostic.file).name}:{diagnostic.line}"
            if diagnostic.python_line is not None:
                location += f" (Python line {diagnostic.python_line})"
            lines.append(f"{idx}. {diagnostic.obligation_kind.title()} failure at {location}")
            lines.append(f"   Message: {diagnostic.message}")
            if diagnostic.call_name:
                lines.append(f"   Related call: {diagnostic.call_name}(...)")
            if diagnostic.failing_text:
                lines.append(f"   Failing code: {diagnostic.failing_text}")
            if diagnostic.related_file and diagnostic.related_line:
                related_location = f"{Path(diagnostic.related_file).name}:{diagnostic.related_line}"
                related_message = diagnostic.related_message or "Related contract location from Dafny"
                lines.append(f"   Related contract: {related_location} ({related_message})")
            if diagnostic.source_excerpt:
                lines.append("   Local code excerpt:")
                lines.extend(f"     {line}" for line in diagnostic.source_excerpt.splitlines())
            if diagnostic.contract_excerpt:
                lines.append("   Relevant contract excerpt:")
                lines.extend(f"     {line}" for line in diagnostic.contract_excerpt.splitlines())

        return "\n".join(lines)


class DafnyVerifier:
    """
    Wrapper for Dafny verification.
    
    Writes generated Python code to a temp workspace, transpiles it to Dafny,
    runs verification, and parses results.
    """

    # Regex patterns for parsing Dafny output
    ERROR_PATTERN = re.compile(
        r"^(.+?)\((\d+),(\d+)\):\s*(Error|Warning|Info):\s*(.+)$",
        re.MULTILINE
    )
    RELATED_PATTERN = re.compile(
        r"^(.+?)\((\d+),(\d+)\):\s*Related location:\s*(.+)$"
    )
    PYTHON_LINE_MARKER_PATTERN = re.compile(r"^\s*//\s*Python line\s+(\d+)\s*$")
    
    def __init__(
        self,
        dafny_path: str = "dafny",
        timeout: int = 60,
        extra_args: Optional[list[str]] = None
    ):
        """
        Initialize the verifier.
        
        Args:
            dafny_path: Path to dafny executable
            timeout: Verification timeout in seconds
            extra_args: Additional arguments to pass to dafny
        """
        self.dafny_path = dafny_path
        self.timeout = timeout
        self.extra_args = extra_args or []
        
        # Verify dafny is available
        check_dafny_available(self.dafny_path)

    def _parse_errors(self, output: str, source_file: str) -> list[VerificationError]:
        """
        Parse verification errors from Dafny output.
        
        Args:
            output: Raw Dafny output
            source_file: Path to the source file being verified
            
        Returns:
            List of parsed errors
        """
        errors = []
        
        for match in self.ERROR_PATTERN.finditer(output):
            file_path = match.group(1)
            line = int(match.group(2))
            column = int(match.group(3))
            error_type = match.group(4)
            message = match.group(5).strip()
            
            # Only include actual errors (not warnings/info)
            if error_type == "Error":
                errors.append(VerificationError(
                    file=file_path,
                    line=line,
                    column=column,
                    message=message,
                    error_type=error_type
                ))
        
        return errors

    def _build_python_line_map(self, dafny_source: str) -> dict[int, int]:
        """Map transpiled Dafny line numbers back to the nearest tagged Python source line."""
        current_python_line: Optional[int] = None
        mapping: dict[int, int] = {}
        for lineno, raw_line in enumerate(dafny_source.splitlines(), start=1):
            marker_match = self.PYTHON_LINE_MARKER_PATTERN.match(raw_line)
            if marker_match:
                current_python_line = int(marker_match.group(1))
                continue
            if current_python_line is not None:
                mapping[lineno] = current_python_line
        return mapping

    def _attach_python_line_numbers(
        self,
        errors: list[VerificationError],
        python_line_map: dict[int, int],
    ) -> list[VerificationError]:
        for error in errors:
            error.python_line = python_line_map.get(error.line)
        return errors

    @staticmethod
    def _classify_obligation(message: str) -> str:
        lowered = message.lower()
        if "loop invariant" in lowered or "invariant could not be proved" in lowered:
            return "invariant"
        if "precondition for this call" in lowered:
            return "precondition"
        if "postcondition could not be proved" in lowered:
            return "postcondition"
        if "decreases expression" in lowered:
            return "decreases"
        return "verification"

    @staticmethod
    def _extract_line(text: str, line_no: int) -> str:
        lines = text.splitlines()
        if 1 <= line_no <= len(lines):
            return lines[line_no - 1].strip()
        return ""

    @staticmethod
    def _extract_excerpt(text: str, line_no: int, radius: int = 2) -> str:
        lines = text.splitlines()
        if not lines or line_no <= 0:
            return ""
        start = max(1, line_no - radius)
        end = min(len(lines), line_no + radius)
        excerpt_lines = []
        for current in range(start, end + 1):
            marker = ">" if current == line_no else " "
            excerpt_lines.append(f"{marker} {current}: {lines[current - 1]}")
        return "\n".join(excerpt_lines)

    @staticmethod
    def _extract_call_name(source_line: str) -> str:
        match = re.search(r"(?:helpers\.|lm\.|parser\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\(", source_line)
        return match.group(1) if match else ""

    def _match_error_blocks(self, output: str, errors: list[VerificationError]) -> list[list[str]]:
        lines = output.splitlines()
        error_indices = [idx for idx, line in enumerate(lines) if self.ERROR_PATTERN.match(line)]
        blocks: list[list[str]] = []
        for pos, _ in enumerate(errors):
            if pos >= len(error_indices):
                blocks.append([])
                continue
            start = error_indices[pos]
            end = error_indices[pos + 1] if pos + 1 < len(error_indices) else len(lines)
            blocks.append(lines[start:end])
        return blocks

    def _build_diagnostics(
        self,
        output: str,
        errors: list[VerificationError],
        generated_source: str,
        proof_source: str,
        python_line_map: dict[int, int],
    ) -> list[VerificationDiagnostic]:
        diagnostics: list[VerificationDiagnostic] = []
        blocks = self._match_error_blocks(output, errors)

        for error, block in zip(errors, blocks):
            source_name = Path(error.file).name
            source_text = proof_source if source_name == "VerifiedAgentSynthesis.dfy" else generated_source
            source_line = self._extract_line(source_text, error.line)
            diagnostic = VerificationDiagnostic(
                file=error.file,
                line=error.line,
                column=error.column,
                message=error.message,
                obligation_kind=self._classify_obligation(error.message),
                failing_text=source_line,
                source_excerpt=self._extract_excerpt(source_text, error.line),
                call_name=self._extract_call_name(source_line),
                python_line=python_line_map.get(error.line) if source_name != "VerifiedAgentSynthesis.dfy" else None,
            )

            for raw_line in block:
                related = self.RELATED_PATTERN.match(raw_line)
                if not related:
                    continue
                diagnostic.related_file = related.group(1)
                diagnostic.related_line = int(related.group(2))
                diagnostic.related_message = related.group(4).strip()
                related_name = Path(diagnostic.related_file).name
                if related_name == "VerifiedAgentSynthesis.dfy":
                    diagnostic.contract_excerpt = self._extract_excerpt(proof_source, diagnostic.related_line)
                elif related_name == "GeneratedCSD.dfy":
                    diagnostic.contract_excerpt = self._extract_excerpt(generated_source, diagnostic.related_line)
                break

            diagnostics.append(diagnostic)

        return diagnostics
    
    def verify(self, python_code: str) -> VerificationResult:
        """
        Verify generated Python strategy code.
        
        Args:
            python_code: Complete Python source code
            
        Returns:
            VerificationResult with success status and any errors
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            try:
                source_file, cwd, _ = prepare_temp_dafny_dir(temp_path, python_code)
                dafny_source = source_file.read_text(encoding="utf-8")
            except Exception as e:
                return VerificationResult(
                    success=False,
                    errors=[VerificationError(
                        file="System", line=0, column=0,
                        message=str(e),
                    )],
                    return_code=-1,
                )

            proof_source_text = ""
            proof_path = get_verified_agent_synthesis_dafny_path()
            if proof_path is not None and proof_path.exists():
                proof_source_text = proof_path.read_text(encoding="utf-8")

            # Run dafny verify
            cmd = [
                self.dafny_path,
                "verify",
                str(source_file),
                *self.extra_args
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=cwd,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    success=False,
                    errors=[VerificationError(
                        file=str(source_file),
                        line=0,
                        column=0,
                        message=f"Verification timed out after {self.timeout} seconds"
                    )],
                    raw_output="",
                    raw_stderr="Timeout",
                    return_code=-1
                )
            
            # Parse the output
            combined_output = result.stdout + result.stderr
            errors = self._parse_errors(combined_output, str(source_file))
            python_line_map = self._build_python_line_map(dafny_source)
            errors = self._attach_python_line_numbers(errors, python_line_map)
            diagnostics = self._build_diagnostics(
                combined_output,
                errors,
                dafny_source,
                proof_source_text,
                python_line_map,
            )
            
            # Dafny returns 0 on success
            success = result.returncode == 0 and len(errors) == 0
            
            return VerificationResult(
                success=success,
                errors=errors,
                raw_output=result.stdout,
                raw_stderr=result.stderr,
                return_code=result.returncode,
                diagnostics=diagnostics,
            )

    def verify_dafny(self, dafny_code: str) -> VerificationResult:
        """Verify generated Dafny strategy code directly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            try:
                source_file, cwd = prepare_temp_dafny_dir_from_dafny(temp_path, dafny_code)
                dafny_source = source_file.read_text(encoding="utf-8")
            except Exception as e:
                return VerificationResult(
                    success=False,
                    errors=[VerificationError(
                        file="System", line=0, column=0,
                        message=str(e),
                    )],
                    return_code=-1,
                )

            proof_source_text = ""
            proof_path = get_verified_agent_synthesis_dafny_path()
            if proof_path is not None and proof_path.exists():
                proof_source_text = proof_path.read_text(encoding="utf-8")

            cmd = [
                self.dafny_path,
                "verify",
                str(source_file),
                *self.extra_args
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=cwd,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    success=False,
                    errors=[VerificationError(
                        file=str(source_file),
                        line=0,
                        column=0,
                        message=f"Verification timed out after {self.timeout} seconds"
                    )],
                    raw_output="",
                    raw_stderr="Timeout",
                    return_code=-1
                )

            combined_output = result.stdout + result.stderr
            errors = self._parse_errors(combined_output, str(source_file))
            python_line_map = self._build_python_line_map(dafny_source)
            errors = self._attach_python_line_numbers(errors, python_line_map)
            diagnostics = self._build_diagnostics(
                combined_output,
                errors,
                dafny_source,
                proof_source_text,
                python_line_map,
            )
            success = result.returncode == 0 and len(errors) == 0
            return VerificationResult(
                success=success,
                errors=errors,
                raw_output=result.stdout,
                raw_stderr=result.stderr,
                return_code=result.returncode,
                diagnostics=diagnostics,
            )
    
    def verify_file(self, file_path: Path) -> VerificationResult:
        """
        Verify a generated Python strategy file directly.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            VerificationResult
        """
        if not file_path.exists():
            return VerificationResult(
                success=False,
                errors=[VerificationError(
                    file=str(file_path),
                    line=0,
                    column=0,
                    message=f"File not found: {file_path}"
                )],
                return_code=-1
            )
        
        return self.verify(file_path.read_text())
