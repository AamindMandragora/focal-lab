"""
Dafny to Python compiler wrapper for CSD synthesis.

Runs `dafny build --target:py` to compile verified Dafny code to Python.
"""

import os
import re
import shutil
import subprocess
import tempfile


def _dafny_subprocess_env() -> dict:
    """Return env for Dafny subprocesses with optional extra PATH entries.

    Configure via `DAFNY_EXTRA_PATH`, a colon-separated list of directories
    prepended to PATH before running Dafny. If unset, we keep the historical
    default `/opt/anaconda/bin` for compatibility with local z3 installs.
    """
    env = os.environ.copy()
    extra = os.environ.get("DAFNY_EXTRA_PATH", "/opt/anaconda/bin")
    extra_entries = [entry for entry in extra.split(":") if entry]
    current = env.get("PATH", "")
    current_entries = current.split(":") if current else []
    prepend_entries = [entry for entry in extra_entries if entry not in current_entries]
    if prepend_entries:
        env["PATH"] = ":".join(prepend_entries + current_entries)
    return env
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import secrets
from typing import Optional

try:
    from synthesis.prompt_rendering import render as _render_prompt
    from synthesis.prompt_rendering.models.verify import (
        CompilationErrorSummaryModel,
        ErrorEntry,
    )
except ImportError:
    from prompt_rendering import render as _render_prompt
    from prompt_rendering.models.verify import (
        CompilationErrorSummaryModel,
        ErrorEntry,
    )


def _resolve_verified_agent_synthesis_path(default_proofs_dir: Path) -> Path:
    """Resolve VerifiedAgentSynthesis.dfy with env-based override support."""
    explicit_file = os.environ.get("VERIFIED_AGENT_SYNTHESIS_DFY")
    if explicit_file:
        return Path(explicit_file).expanduser()

    proofs_dir_override = os.environ.get("DAFNY_PROOFS_DIR")
    if proofs_dir_override:
        return Path(proofs_dir_override).expanduser() / "VerifiedAgentSynthesis.dfy"

    return default_proofs_dir / "VerifiedAgentSynthesis.dfy"


@dataclass
class CompilationError:
    """A single compilation error from Dafny."""
    file: str
    line: int
    column: int
    message: str
    
    def __str__(self) -> str:
        return f"{self.file}({self.line},{self.column}): {self.message}"


@dataclass
class CompilationResult:
    """Result of Dafny to Python compilation."""
    success: bool
    output_dir: Optional[Path] = None
    main_module_path: Optional[Path] = None
    errors: list[CompilationError] = field(default_factory=list)
    raw_output: str = ""
    raw_stderr: str = ""
    return_code: int = 0
    
    def get_error_summary(self) -> str:
        """Get a human-readable summary of errors."""
        if self.success:
            return f"Compilation successful. Output: {self.output_dir}"

        if not self.errors:
            return self.raw_output or self.raw_stderr

        model = CompilationErrorSummaryModel(
            error_count=len(self.errors),
            errors=[ErrorEntry(line=err.line, message=err.message) for err in self.errors],
        )
        rendered = _render_prompt(model, "verify/compilation_error_summary.j2")
        # The original implementation joined its lines with "\n" and never
        # added a trailing newline; keep_trailing_newline on the shared Jinja
        # environment means the template's own final line terminator survives
        # rendering, so strip exactly that one trailing newline back off (same
        # idiom as EvaluationResult.get_feedback_summary).
        if rendered.endswith("\n"):
            rendered = rendered[:-1]
        return rendered


class DafnyCompiler:
    """
    Wrapper for Dafny to Python compilation.
    
    Compiles verified Dafny code to Python modules that can be imported and run.
    """
    
    # Path to proofs directory (for includes)
    PROOFS_DIR = Path(__file__).parent.parent.parent / "proofs"
    
    # Default output directory (run-specific compiler directories are set by the pipeline).
    DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "generated"
    
    # Regex pattern for parsing Dafny errors
    ERROR_PATTERN = re.compile(
        r"^(.+?)\((\d+),(\d+)\):\s*(?:Error|error):\s*(.+)$",
        re.MULTILINE
    )
    
    def __init__(
        self,
        dafny_path: str = "dafny",
        output_dir: Optional[Path] = None,
        timeout: int = 120,
        extra_args: Optional[list[str]] = None
    ):
        """
        Initialize the compiler.
        
        Args:
            dafny_path: Path to dafny executable
            output_dir: Directory to store compiled Python code
            timeout: Compilation timeout in seconds
            extra_args: Additional arguments to pass to dafny
        """
        self.dafny_path = dafny_path
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self.timeout = timeout
        self.extra_args = extra_args or []
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_errors(self, output: str) -> list[CompilationError]:
        """Parse compilation errors from Dafny output."""
        errors = []
        
        for match in self.ERROR_PATTERN.finditer(output):
            errors.append(CompilationError(
                file=match.group(1),
                line=int(match.group(2)),
                column=int(match.group(3)),
                message=match.group(4).strip()
            ))
        
        return errors

    def _unique_final_output_dir(self, output_name: str) -> Path:
        """
        Pick a final output directory that won't clobber an existing compilation.

        Historically we wrote to `<output_dir>/<output_name>` and deleted it if it
        already existed. That made concurrent / repeated runs overwrite each other.
        """
        base = self.output_dir / output_name
        if not base.exists():
            return base

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = secrets.token_hex(3)
        return self.output_dir / f"{output_name}_{ts}_{suffix}"
    
    def compile(
        self,
        dafny_code: str,
        output_name: str = "generated_csd"
    ) -> CompilationResult:
        """
        Compile Dafny code to Python.
        
        Args:
            dafny_code: Complete Dafny source code (should be verified first)
            output_name: Name for the output module
            
        Returns:
            CompilationResult with paths to generated Python code
        """
        # Create temp directory for compilation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure matching includes
            agents_dir = temp_path / "agents"
            agents_dir.mkdir()
            
            proofs_dir = temp_path / "proofs"
            proofs_dir.mkdir()
            
            # Copy the VerifiedAgentSynthesis.dfy
            source_proof = _resolve_verified_agent_synthesis_path(self.PROOFS_DIR)
            # Fallback: check library/ directory if not in proofs/
            if not source_proof.exists():
                source_proof = Path(__file__).parent / "library" / "VerifiedAgentSynthesis.dfy"

            if source_proof.exists():
                (proofs_dir / "VerifiedAgentSynthesis.dfy").write_text(
                    source_proof.read_text()
                )

                # Copy into agents/ directory too, as some relative imports might look there
                (agents_dir / "VerifiedAgentSynthesis.dfy").write_text(
                    source_proof.read_text()
                )
            
            # Write the generated code
            source_file = agents_dir / "GeneratedCSD.dfy"
            source_file.write_text(dafny_code)
            
            # Output directory within temp (Dafny creates a subdirectory)
            compile_output = temp_path / "compiled"
            compile_output.mkdir()
            
            # Run dafny build with Python target
            cmd = [
                self.dafny_path,
                "build",
                "--target:py",
                f"--output:{compile_output / output_name}",
                str(source_file),
                *self.extra_args
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=temp_path,
                    env=_dafny_subprocess_env()
                )
            except subprocess.TimeoutExpired:
                return CompilationResult(
                    success=False,
                    errors=[CompilationError(
                        file=str(source_file),
                        line=0,
                        column=0,
                        message=f"Compilation timed out after {self.timeout} seconds"
                    )],
                    return_code=-1
                )
            
            # Check for errors
            combined_output = result.stdout + result.stderr
            errors = self._parse_errors(combined_output)
            
            if result.returncode != 0 or errors:
                return CompilationResult(
                    success=False,
                    errors=errors,
                    raw_output=result.stdout,
                    raw_stderr=result.stderr,
                    return_code=result.returncode
                )
            
            # Find the generated Python files
            # Dafny creates a directory with the module name
            generated_dir = compile_output / output_name
            
            # If the directory doesn't exist, check for .py files directly
            if not generated_dir.exists():
                # Look for any .py files
                py_files = list(compile_output.glob("**/*.py"))
                if not py_files:
                    return CompilationResult(
                        success=False,
                        errors=[CompilationError(
                            file=str(source_file),
                            line=0,
                            column=0,
                            message="Compilation succeeded but no Python files were generated"
                        )],
                        raw_output=result.stdout,
                        raw_stderr=result.stderr,
                        return_code=result.returncode
                    )
                # Use the first .py file's parent as the generated dir
                generated_dir = py_files[0].parent
            
            # Copy generated files to output directory
            final_output_dir = self._unique_final_output_dir(output_name)
            shutil.copytree(generated_dir, final_output_dir)
            
            # Find the main module file
            main_module = None
            for py_file in final_output_dir.glob("**/*.py"):
                if py_file.stem == output_name or py_file.stem == "GeneratedCSD":
                    main_module = py_file
                    break
            
            # If no specific match, use any .py file
            if main_module is None:
                py_files = list(final_output_dir.glob("**/*.py"))
                if py_files:
                    main_module = py_files[0]
            
            return CompilationResult(
                success=True,
                output_dir=final_output_dir,
                main_module_path=main_module,
                raw_output=result.stdout,
                raw_stderr=result.stderr,
                return_code=result.returncode
            )
    
    def compile_file(
        self,
        file_path: Path,
        output_name: Optional[str] = None
    ) -> CompilationResult:
        """
        Compile a Dafny file to Python.
        
        Args:
            file_path: Path to the Dafny file
            output_name: Name for the output module (default: file stem)
            
        Returns:
            CompilationResult
        """
        if not file_path.exists():
            return CompilationResult(
                success=False,
                errors=[CompilationError(
                    file=str(file_path),
                    line=0,
                    column=0,
                    message=f"File not found: {file_path}"
                )],
                return_code=-1
            )
        
        output_name = output_name or file_path.stem
        return self.compile(file_path.read_text(), output_name)
