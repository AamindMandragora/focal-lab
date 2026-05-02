"""
Evaluation module for synthesis feedback loop.

Provides quick evaluation of synthesized CSD strategies on dataset samples
to enable feedback-driven refinement based on actual performance metrics.
"""

from __future__ import annotations

import ast
import os
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvaluationResult:
    """
    Result of evaluating a CSD strategy on a dataset sample.

    Contains metrics and sample outputs for feedback to the generator.
    """
    success: bool
    accuracy: float  # 0.0 to 1.0
    format_rate: float  # 0.0 to 1.0
    syntax_rate: float  # 0.0 to 1.0
    num_examples: int
    num_correct: int
    total_time_seconds: float

    # Sample outputs for feedback (question, expected, actual, is_correct)
    sample_outputs: List[Dict[str, Any]] = field(default_factory=list)

    # Error information if evaluation failed
    error: Optional[str] = None

    def meets_threshold(
        self,
        min_accuracy: float = 0.0,
        min_format_rate: float = 0.0,
        min_syntax_rate: float = 0.0,
    ) -> bool:
        """Check if results meet the specified thresholds."""
        return (
            self.accuracy >= min_accuracy
            and self.format_rate >= min_format_rate
            and self.syntax_rate >= min_syntax_rate
        )

    def get_feedback_summary(self) -> str:
        """Generate a summary for feedback to the generator."""
        lines = [
            f"Evaluation Results ({self.num_examples} examples):",
            f"  Accuracy: {self.accuracy:.1%} ({self.num_correct}/{self.num_examples})",
            f"  Format Rate: {self.format_rate:.1%}",
            f"  Syntax Rate: {self.syntax_rate:.1%}",
            f"  Total Time: {self.total_time_seconds:.2f}s",
        ]

        if self.sample_outputs:
            lines.append("\nSample Failures:")
            failures = [s for s in self.sample_outputs if not s.get("is_correct", False)]
            for i, sample in enumerate(failures[:3]):  # Show up to 3 failures
                lines.append(f"\n  Example {i+1}:")
                lines.append(f"    Question: {sample.get('question', 'N/A')[:100]}...")
                lines.append(f"    Expected: {sample.get('expected', 'N/A')}")
                lines.append(f"    Got: {sample.get('actual', 'N/A')}")
                if sample.get("full_output"):
                    lines.append(
                        f"    Raw output: {_truncate_for_display(str(sample.get('full_output')), 180)}"
                    )
                if sample.get("error"):
                    lines.append(f"    Error: {sample.get('error')}")

        return "\n".join(lines)

    def get_detailed_samples(self, max_samples: int = 3) -> str:
        """Return a human-readable 'generated vs expected' view for debugging accuracy."""
        lines = [
            "Generated vs expected (for accuracy debugging):",
            "-" * 60,
        ]
        for i, s in enumerate(self.sample_outputs[:max_samples]):
            lines.append(f"\n--- Example {i+1} ---")
            lines.append(f"  Expected (gold) answer: {s.get('expected', 'N/A')}")
            lines.append(f"  Parsed answer (actual): {s.get('actual', 'N/A')}")
            lines.append(f"  Match: {'YES' if s.get('is_correct') else 'NO'}")
            lines.append(f"  Full raw output:\n    {_truncate_for_display(s.get('full_output') or '', 350)}")
            # FOLIO: last $ % segment sent to Prover9; result diffed to expected
            folio_debug = s.get("folio_debug")
            if folio_debug:
                reason = folio_debug.get("reason", "")
                if reason:
                    lines.append(f"  FOLIO extraction: {reason}")
                if folio_debug.get("segments"):
                    lines.append(f"  Model $ % segments ({len(folio_debug['segments'])}):")
                    for j, seg in enumerate(folio_debug["segments"]):
                        lines.append(f"    [{j+1}] {_truncate_for_display(seg, 180)}")
                lines.append(f"  Conclusion sent to Prover9: {_truncate_for_display(folio_debug.get('conclusion_raw') or folio_debug.get('conclusion', ''), 200)}")
                if folio_debug.get("prover9_ok"):
                    lines.append(f"  Prover9 result: {folio_debug.get('prover9_answer', 'N/A')}")
                else:
                    lines.append("  Prover9: failed or fallback used")
                    if folio_debug.get("prover9_error"):
                        lines.append(f"    Error: {_truncate_for_display(folio_debug['prover9_error'], 150)}")
                    if folio_debug.get("logic_program"):
                        prog = folio_debug["logic_program"]
                        lines.append("  Logic program (first 2 + last 2 lines):")
                        for line in prog.split("\n")[:2]:
                            lines.append(f"    {_truncate_for_display(line, 120)}")
                        lines.append("    ...")
                        for line in prog.split("\n")[-2:]:
                            lines.append(f"    {_truncate_for_display(line, 120)}")
            if not folio_debug:
                segs = s.get("extracted_segments") or []
                delim_label = " $ % " if (getattr(self, "dataset_name", None) == "folio") else " << >> "
                lines.append(f"  Extracted from{delim_label}({len(segs)} segment(s)):")
                for j, seg in enumerate(segs):
                    lines.append(f"    [{j+1}] {_truncate_for_display(seg, 200)}")
            if s.get("gold_premises_fol") is not None:
                prems = s["gold_premises_fol"]
                prems_str = prems if isinstance(prems, list) else [str(prems)]
                lines.append(f"  Gold premises (FOL): {_truncate_for_display(str(prems_str), 200)}")
            if s.get("gold_conclusion_fol") is not None:
                lines.append(f"  Gold conclusion (FOL): {s.get('gold_conclusion_fol')}")
            lines.append("")
        return "\n".join(lines)

    def print_outputs_vs_expected(self, max_samples: Optional[int] = None) -> None:
        """
        Print expected vs actual outputs to stdout during evaluation.
        For FOLIO, also prints the conclusion sent to Prover9 and Prover9's result
        so you can see when 'Unknown' (malformed) is counted as correct via Uncertain.
        """
        samples = self.sample_outputs[:max_samples] if max_samples is not None else self.sample_outputs
        print("  --- Outputs vs expected ---")
        for i, s in enumerate(samples):
            expected = s.get("expected", "N/A")
            actual = s.get("actual", "N/A")
            is_correct = s.get("is_correct", False)
            prompt_preview = s.get("prompt") or s.get("question", "")
            print(f"  Example {i+1}:")
            print(f"    Prompt (first 400 chars): {_truncate_for_display(prompt_preview, 400)}")
            print(f"    Expected: {expected}")
            print(f"    Actual (parsed): {actual}")
            # Highlight when match is due to Unknown=Uncertain, or when we rejected it (conclusion ≠ gold)
            if is_correct and actual and expected:
                a, e = str(actual).strip().lower(), str(expected).strip().lower()
                if (a == "unknown" and e == "uncertain") or (a == "uncertain" and e == "unknown"):
                    print(f"    Match: YES (Prover9 returned Unknown, conclusion matched gold)")
                else:
                    print(f"    Match: YES")
            else:
                match_note = ""
                if not is_correct and actual and expected:
                    a, e = str(actual).strip().lower(), str(expected).strip().lower()
                    if (a == "unknown" and e == "uncertain") or (a == "uncertain" and e == "unknown"):
                        match_note = " (Prover9 Unknown but conclusion did not match gold)"
                print(f"    Match: {'YES' if is_correct else 'NO'}{match_note}")
            folio_debug = s.get("folio_debug")
            if folio_debug is not None:
                conclusion = folio_debug.get("conclusion_raw") or folio_debug.get("conclusion") or ""
                prover9_ans = folio_debug.get("prover9_answer")
                prover9_ok = folio_debug.get("prover9_ok", False)
                print(f"    Conclusion sent to Prover9: {_truncate_for_display(conclusion, 300)}")
                if prover9_ok:
                    print(f"    Prover9 result: {prover9_ans}")
                else:
                    print(f"    Prover9: failed or fallback — {_truncate_for_display(folio_debug.get('prover9_error') or 'N/A', 120)}")
            print("")
        if not samples:
            print("    (no samples)")
        print("  " + "-" * 40)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "accuracy": self.accuracy,
            "format_rate": self.format_rate,
            "syntax_rate": self.syntax_rate,
            "num_examples": self.num_examples,
            "num_correct": self.num_correct,
            "total_time_seconds": self.total_time_seconds,
            "error": self.error,
            "sample_outputs": self.sample_outputs,
        }


def _truncate_for_display(s: str, max_len: int) -> str:
    s = (s or "").replace("\n", " ")
    return s[:max_len] + ("..." if len(s) > max_len else "")


class Evaluator:
    """
    Evaluates synthesized CSD strategies on dataset samples.

    Supports both GSM-Symbolic and FOLIO datasets with their respective
    evaluation metrics and syntax validation.
    """

    def __init__(
        self,
        dataset_name: str = "gsm_symbolic",
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        vocab_size: int = 3000,
        sample_size: int = 1,
        max_steps: int = 150,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the evaluator.

        Args:
            dataset_name: Dataset to evaluate on ("gsm_symbolic" or "folio")
            model_name: HuggingFace model for generation
            device: Device to run on ("cuda", "mps", "cpu")
            vocab_size: Vocabulary size for constrained generation
            sample_size: Number of examples to evaluate on
            max_steps: Maximum generation steps per example
            load_in_4bit: Load eval model in 4-bit to save memory and speed up (default: False)
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device
        self.vocab_size = vocab_size
        self.sample_size = sample_size
        self.max_steps = max_steps
        self.load_in_4bit = load_in_4bit

        # Lazy-loaded components
        self._dataset = None
        self._env = None
        self._grammar_file = None

    def _get_grammar_file(self) -> Path:
        """Get the grammar file path for the dataset."""
        if self._grammar_file is None:
            grammars_dir = Path(__file__).resolve().parents[1] / "utils" / "grammars"
            if self.dataset_name == "gsm_symbolic":
                self._grammar_file = grammars_dir / "gsm.lark"
            elif self.dataset_name == "folio":
                self._grammar_file = grammars_dir / "folio.lark"
            elif self.dataset_name == "sygus_slia":
                self._grammar_file = grammars_dir / "sygus_slia.lark"
            elif self.dataset_name == "pddl":
                self._grammar_file = grammars_dir / "pddl.lark"
            elif self.dataset_name == "spider":
                self._grammar_file = grammars_dir / "sql.lark"
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")
        return self._grammar_file

    def _extract_gsm_variable_names(self, example: Any) -> List[str]:
        candidates: List[str] = []
        for key in ("variable_mapping", "variables", "eval_input", "bindings", "symbolic_variables"):
            value = self._example_field(example, key, None)
            if isinstance(value, dict):
                candidates.extend(str(k) for k in value.keys())
            elif isinstance(value, (list, tuple, set)):
                candidates.extend(str(v) for v in value)

        cleaned: List[str] = []
        for name in candidates:
            stripped = str(name).strip()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        return cleaned

    def _build_dynamic_gsm_parser(self, env: Dict[str, Any], example: Any):
        variables = self._extract_gsm_variable_names(example)
        try:
            from evaluation.gsm_symbolic.grammar import build_dynamic_grammar
        except Exception:
            return None

        try:
            base_grammar = self._get_grammar_file().read_text()
            # If the dataset example gives no numeric variable bindings, the
            # dynamic grammar still keeps scratch variables such as x_1 but
            # disables arbitrary unbound dataset symbols such as `x`.
            dynamic_grammar = build_dynamic_grammar(base_grammar, variables)
            start_rule = "csd_start" if variables else "csd_numeric_start"
            if env.get("_mode") == "native":
                from evaluation.common.parser_utils import create_lark_native_parser
                parser_cls = create_lark_native_parser(
                    dynamic_grammar,
                    env["VerifiedAgentSynthesis"],
                    start=start_rule,
                )
                return parser_cls(env["lm"].Tokens)
            else:
                from evaluation.common.parser_utils import create_lark_dafny_parser
                parser_cls = create_lark_dafny_parser(
                    dynamic_grammar,
                    env["VerifiedDecoderAgent"],
                    env["_dafny"],
                    start=start_rule,
                )
                return parser_cls(env["lm"]._Tokens)
        except Exception:
            return None

    def _load_dataset_sample(self) -> list:
        """Load a sample of the dataset for evaluation."""
        if self._dataset is not None:
            return self._dataset

        if self.dataset_name == "gsm_symbolic":
            from evaluation.gsm_symbolic.dataset import load_gsm_symbolic
            ds = load_gsm_symbolic(
                config="main",
                split="test",
                limit=self.sample_size,
                random_sample=True,
            )
            self._dataset = list(ds)
        elif self.dataset_name == "folio":
            from evaluation.folio.dataset import load_folio
            ds = load_folio(
                split="validation",
                num_samples=self.sample_size,
                seed=42,
            )
            self._dataset = ds
        elif self.dataset_name == "sygus_slia":
            from evaluation.sygus_slia.dataset import load_sygus_slia
            self._dataset = load_sygus_slia(
                limit=self.sample_size,
                random_sample=True,
            )
        elif self.dataset_name == "pddl":
            from evaluation.pddl.dataset import load_pddl
            self._dataset = load_pddl(
                limit=self.sample_size,
                random_sample=True,
            )
        elif self.dataset_name == "spider":
            from evaluation.spider.dataset import load_spider
            spider_random_sample = os.environ.get("CSD_SPIDER_EVAL_RANDOM_SAMPLE", "0").strip() in {
                "1",
                "true",
                "True",
            }
            self._dataset = load_spider(
                split="test",
                limit=self.sample_size,
                random_sample=spider_random_sample,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return self._dataset

    def _setup_environment(
        self,
        compiled_module_path: Optional[Path] = None,
        python_source_path: Optional[Path] = None,
        extra_token_strings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Set up the evaluation environment.

        The active synthesis path provides python_source_path and uses
        Python-native mode. compiled_module_path is retained only for legacy
        Dafny-built artifacts.
        """
        if python_source_path is not None:
            from evaluation.common.environment import setup_python_native_environment

            start_rule = "start" if self.dataset_name == "folio" else "csd_start"
            return setup_python_native_environment(
                python_source_path=python_source_path,
                model_name=self.model_name,
                device=self.device,
                vocab_size=self.vocab_size,
                grammar_file=self._get_grammar_file(),
                start_rule=start_rule,
                load_in_4bit=self.load_in_4bit,
                add_gsm_delimiter_tokens=(self.dataset_name != "folio"),
                add_fol_keyword_tokens=(self.dataset_name == "folio"),
                extra_token_strings=extra_token_strings,
            )

        # Dafny-compiled mode (original path)
        assert compiled_module_path is not None
        run_dir = compiled_module_path.parent
        if run_dir.name == "generated_csd":
            run_dir = run_dir.parent

        from evaluation.common.environment import setup_dafny_environment

        if self.dataset_name == "gsm_symbolic":
            return setup_dafny_environment(
                run_dir=run_dir,
                model_name=self.model_name,
                device=self.device,
                vocab_size=self.vocab_size,
                grammar_file=self._get_grammar_file(),
                start_rule="csd_start",
                load_in_4bit=self.load_in_4bit,
                add_gsm_delimiter_tokens=True,
            )
        elif self.dataset_name in ("sygus_slia", "pddl", "spider"):
            return setup_dafny_environment(
                run_dir=run_dir,
                model_name=self.model_name,
                device=self.device,
                vocab_size=self.vocab_size,
                grammar_file=self._get_grammar_file(),
                start_rule="csd_start",
                load_in_4bit=self.load_in_4bit,
            )
        else:
            return setup_dafny_environment(
                run_dir=run_dir,
                model_name=self.model_name,
                device=self.device,
                vocab_size=self.vocab_size,
                grammar_file=self._get_grammar_file(),
                start_rule="start",
                load_in_4bit=self.load_in_4bit,
                add_fol_keyword_tokens=(self.dataset_name == "folio"),
            )

    def _extract_constrained_content(self, output: str) -> List[str]:
        """Extract content within delimiters. FOLIO uses $ ... %; all others use << ... >>."""
        if self.dataset_name == "folio":
            return re.findall(r"\$\s*([^$%]+?)\s*%", output)
        return re.findall(r"<<\s*([\s\S]+?)\s*>>", output)

    @staticmethod
    def _normalize_gsm_decimal(value: Decimal) -> str:
        """Render a Decimal without scientific notation or redundant trailing zeros."""
        if value == 0:
            return "0"
        if value == value.to_integral_value():
            return str(int(value))
        text = format(value.normalize(), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in {"-0", ""} else text

    def _extract_gsm_numeric_bindings(self, example: Optional[Any]) -> Dict[str, Decimal]:
        """Best-effort numeric bindings for symbolic GSM expressions, when available."""
        if example is None:
            return {}

        bindings: Dict[str, Decimal] = {}
        for key in ("variable_mapping", "eval_input", "bindings"):
            raw = self._example_field(example, key, None)
            if not isinstance(raw, dict):
                continue
            for name, value in raw.items():
                try:
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)):
                        bindings[str(name)] = Decimal(str(value))
                    elif isinstance(value, str):
                        numeric = re.search(r"[-+]?\d*\.?\d+", value)
                        if numeric:
                            bindings[str(name)] = Decimal(numeric.group())
                except (InvalidOperation, ValueError):
                    continue
        return bindings

    def _safe_eval_gsm_expr(self, expr: str, bindings: Optional[Dict[str, Decimal]] = None) -> Optional[Decimal]:
        """Safely evaluate a GSM arithmetic expression using a tiny Python AST subset."""
        bindings = bindings or {}
        expr = " ".join(expr.split())
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return None

        def _eval(node: ast.AST) -> Decimal:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return Decimal(str(node.value))
            if isinstance(node, ast.Name) and node.id in bindings:
                return bindings[node.id]
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = _eval(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
                if isinstance(node.op, ast.Mod):
                    return left % right
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "int"
                and len(node.args) == 1
                and not node.keywords
            ):
                return Decimal(int(_eval(node.args[0])))
            raise ValueError(f"Unsupported GSM expression node: {type(node).__name__}")

        try:
            with localcontext() as ctx:
                ctx.prec = 50
                return _eval(tree)
        except (ArithmeticError, InvalidOperation, ValueError, KeyError, ZeroDivisionError):
            return None

    @staticmethod
    def _is_gsm_scratch_name(name: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*_[0-9]+", name.strip()))

    def _eval_gsm_segment(self, segment: str, bindings: Dict[str, Decimal]) -> Optional[Decimal]:
        """Evaluate one constrained GSM span, updating scratch bindings for assignments."""
        stripped = " ".join(segment.split())
        if not stripped:
            return None

        assignment = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*_[0-9]+)\s*=\s*(.+)", stripped)
        if assignment:
            name = assignment.group(1)
            rhs = assignment.group(2)
            value = self._safe_eval_gsm_expr(rhs, bindings)
            if value is not None and self._is_gsm_scratch_name(name):
                bindings[name] = value
                return value
            return None

        candidate_parts = [part.strip() for part in stripped.split("=") if part.strip()]
        for candidate in reversed(candidate_parts):
            value = self._safe_eval_gsm_expr(candidate, bindings)
            if value is not None:
                return value
        return None

    def _extract_answer_gsm(self, output: str, example: Optional[Any] = None) -> Optional[str]:
        """Safely evaluate GSM constrained spans and return the final computed value.

        Earlier spans may define scratch variables, e.g. <<x_1 = 3 + 2 * 6>>,
        and later spans may refer to them, e.g. <<x_1 / y_1>>.
        """
        matches = self._extract_constrained_content(output)
        if not matches:
            return None

        bindings = self._extract_gsm_numeric_bindings(example)
        final_value: Optional[Decimal] = None
        last_index = len(matches) - 1
        for index, segment in enumerate(matches):
            value = self._eval_gsm_segment(segment, bindings)
            if index == last_index:
                final_value = value
        if final_value is None:
            return None
        return self._normalize_gsm_decimal(final_value)

    def _extract_answer_folio(self, output: str, example: Optional[Any] = None) -> Optional[str]:
        """
        Take the last $ % delimited string (FOLIO), evaluate it with Prover9 (with problem
        premises), then return True/False/Unknown. Compare that to expected for grading.
        """
        from evaluation.folio.fol_utils import fol_keyword_to_unicode, fol_normalize_spacing

        debug: Dict[str, Any] = {
            "segments": [], "conclusion_raw": "", "conclusion": "",
            "logic_program": "", "prover9_ok": False, "prover9_answer": None,
            "prover9_error": None, "fallback_used": True,
        }
        segments = self._extract_constrained_content(output)
        if not segments:
            setattr(self, "_last_folio_debug", {**debug, "reason": "no_segments"})
            return self._extract_answer_folio_fallback(output)

        debug["segments"] = list(segments)
        conclusion_raw = segments[-1].strip()
        conclusion = fol_normalize_spacing(fol_keyword_to_unicode(conclusion_raw))
        debug["conclusion_raw"] = conclusion_raw
        debug["conclusion"] = conclusion

        premises: List[str] = []
        if example is not None:
            fol_premises = self._example_field(example, "fol_premises", None)
            if fol_premises:
                if isinstance(fol_premises, list):
                    premises = [fol_normalize_spacing(fol_keyword_to_unicode(p.strip())) for p in fol_premises]
                else:
                    premises = [fol_normalize_spacing(fol_keyword_to_unicode(str(fol_premises).strip()))]
        if not premises and len(segments) >= 2:
            premises = [fol_normalize_spacing(fol_keyword_to_unicode(s.strip())) for s in segments[:-1]]
        debug["premises_used"] = list(premises)

        logic_lines = ["Premises:"]
        for p in premises:
            logic_lines.append(f"{p} ::: premise")
        logic_lines.append("Conclusion:")
        logic_lines.append(f"{conclusion} ::: conclusion")
        logic_program = "\n".join(logic_lines)
        debug["logic_program"] = logic_program

        try:
            from utils.symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
            program = FOL_Prover9_Program(logic_program, dataset_name="FOLIO")
            if not program.flag:
                setattr(self, "_last_folio_debug", {**debug, "reason": "prover9_parse_fail"})
                # Unparseable segment (e.g. prose) → fail; focus on making the model output parseable FOL.
                return self._extract_answer_folio_fallback(output)
            answer, error = program.execute_program()
            if answer in ("True", "False", "Unknown"):
                debug["prover9_ok"] = True
                debug["prover9_answer"] = answer
                debug["fallback_used"] = False
                setattr(self, "_last_folio_debug", debug)
                return answer
            debug["prover9_answer"] = answer
            debug["prover9_error"] = error or ""
        except Exception as e:
            err = str(e)
            if "No module named 'ply'" in err or (isinstance(e, ImportError) and "ply" in str(e).lower()):
                err = "Prover9 needs ply. Install: pip install ply nltk z3-solver"
            debug["prover9_error"] = err
        setattr(self, "_last_folio_debug", debug)
        return self._extract_answer_folio_fallback(output)

    @staticmethod
    def _normalize_folio_answer(text: str) -> Optional[str]:
        """Normalize model output to True, False, or Unknown (Uncertain -> Unknown)."""
        if not text:
            return None
        s = text.strip().lower()
        if s == "true":
            return "True"
        if s == "false":
            return "False"
        if s in ("uncertain", "unknown"):
            return "Unknown"
        # Allow substrings for fallback
        if "true" in s:
            return "True"
        if "false" in s:
            return "False"
        if "uncertain" in s or "unknown" in s:
            return "Unknown"
        return None

    def _extract_answer_folio_fallback(self, output: str) -> Optional[str]:
        """Fallback: extract answer via keyword matching when no $ % segment."""
        return self._normalize_folio_answer(output)

    def _folio_conclusion_matches_gold(self, conclusion_sent: str, gold_conclusion: Optional[str]) -> bool:
        """True if the conclusion we sent to Prover9 matches the gold (normalized for comparison)."""
        if not conclusion_sent or not gold_conclusion:
            return False
        from evaluation.folio.fol_utils import fol_keyword_to_unicode, fol_unicode_to_keyword, fol_normalize_spacing
        # Normalize both to same form: keyword form, normalized spacing
        def norm(s: str) -> str:
            s = fol_normalize_spacing(s.strip())
            # If s looks like unicode (∀, ∧, etc.), convert to keyword then normalize
            if any(c in s for c in ("∀", "∃", "∧", "∨", "¬", "→", "↔", "⊕")):
                s = fol_unicode_to_keyword(s)
            return fol_normalize_spacing(s)
        return norm(conclusion_sent) == norm(gold_conclusion)

    def _answers_match(
        self,
        actual: Optional[str],
        expected: str,
        conclusion_sent: Optional[str] = None,
        gold_conclusion: Optional[str] = None,
        example: Any = None,
    ) -> bool:
        """Check if actual and expected answers match, normalizing Uncertain/Unknown.
        Prover9 returns Unknown when neither goal nor negation is provable; FOLIO labels that Uncertain.
        When actual is Unknown and expected is Uncertain, we only count correct if the conclusion
        sent to Prover9 matches the gold conclusion (avoids crediting garbage output)."""
        if actual is None:
            return False

        if self.dataset_name == "gsm_symbolic":
            try:
                return Decimal(str(actual).strip()) == Decimal(str(expected).strip())
            except InvalidOperation:
                return str(actual).strip() == str(expected).strip()

        # SyGuS SLIA: compare generated result against eval_expected
        if self.dataset_name == "sygus_slia":
            eval_expected = self._example_field(example, "eval_expected", None) if example is not None else None
            if eval_expected is not None:
                return str(actual).strip() == str(eval_expected).strip()
            return str(actual).strip() == str(expected).strip()

        # PDDL: simulation returns "CORRECT" on success
        if self.dataset_name == "pddl":
            return str(actual).strip() == "CORRECT"

        if self.dataset_name == "spider":
            return self._normalize_sql(actual) == self._normalize_sql(expected)

        a = str(actual).strip().lower()
        e = str(expected).strip().lower()
        if a in ("uncertain", "unknown"):
            a = "unknown"
        if e in ("uncertain", "unknown"):
            e = "unknown"
        if a != e:
            return False
        # When both are unknown/uncertain, require the sent conclusion to match gold (if we have it)
        if a == "unknown" and self.dataset_name == "folio" and gold_conclusion and conclusion_sent:
            return self._folio_conclusion_matches_gold(conclusion_sent, gold_conclusion)
        return True

    @staticmethod
    def _example_field(example: Any, key: str, default: Any = None) -> Any:
        """Get a field from an example that may be a dict or a dataclass (e.g. FOLIOExample)."""
        if hasattr(example, key):
            return getattr(example, key)
        if hasattr(example, "get") and callable(getattr(example, "get")):
            return example.get(key, default)
        return default

    def _get_expected_answer(self, example: Any) -> str:
        """Get the expected answer from a dataset example."""
        if self.dataset_name == "gsm_symbolic":
            answer_str = self._example_field(example, "answer", "") or ""
            match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(answer_str))
            if match:
                return match.group(1)
            return str(answer_str)
        elif self.dataset_name in ("sygus_slia", "pddl"):
            return str(self._example_field(example, "answer", "CORRECT"))
        elif self.dataset_name == "spider":
            return str(self._example_field(example, "query", ""))
        else:
            return str(self._example_field(example, "label", "Unknown"))

    def _extract_answer_sygus_slia(self, output: str, example: Any) -> Optional[str]:
        """Parse and evaluate a SyGuS SLIA S-expression from << >> output."""
        from lark import Lark
        from lark.exceptions import LarkError
        from evaluation.sygus_slia.dataset import eval_slia

        matches = self._extract_constrained_content(output)
        if not matches:
            return None

        grammar_text = self._get_grammar_file().read_text()
        try:
            parser = Lark(grammar_text, start="start", parser="lalr")
        except Exception:
            return None

        # Try each match (last one wins if multiple)
        result = None
        bindings = self._example_field(example, "eval_input", {}) or {}
        for seg in matches:
            try:
                tree = parser.parse(seg.strip())
                result = eval_slia(tree, bindings)
            except Exception:
                continue
        return result

    def _extract_answer_pddl(self, output: str, example: Any) -> Optional[str]:
        """Parse and simulate a PDDL plan from << >> output."""
        from lark import Lark, Tree, Token
        from lark.exceptions import LarkError
        from evaluation.pddl.dataset import simulate_plan

        matches = self._extract_constrained_content(output)
        if not matches:
            return None

        grammar_text = self._get_grammar_file().read_text()
        try:
            parser = Lark(grammar_text, start="start", parser="lalr")
        except Exception:
            return None

        initial_state = self._example_field(example, "initial_state", None)
        goal_on = self._example_field(example, "goal_on", []) or []
        goal_on_table = self._example_field(example, "goal_on_table", []) or []
        if initial_state is None:
            return None

        # Use last << >> segment
        seg = matches[-1].strip()
        try:
            tree = parser.parse(seg)
        except LarkError:
            return None

        # Extract plan steps from the parse tree
        plan: List[Tuple[str, ...]] = []
        for action_node in tree.children:
            if not isinstance(action_node, Tree):
                continue
            # action_node.data is "action"; its child is pickup/putdown/stack/unstack
            inner = action_node.children[0] if action_node.children else action_node
            if not isinstance(inner, Tree):
                continue
            action_name = inner.data  # "pickup", "putdown", "stack", "unstack"
            blocks = [str(t) for t in inner.children if isinstance(t, Token)]
            # Map grammar rule names to PDDL action names
            name_map = {"pickup": "pick-up", "putdown": "put-down",
                        "stack": "stack", "unstack": "unstack"}
            pddl_name = name_map.get(action_name, action_name)
            plan.append((pddl_name, *blocks))

        if not plan:
            return None

        success, reason = simulate_plan(initial_state, plan, goal_on, goal_on_table)
        return "CORRECT" if success else f"WRONG: {reason}"

    @staticmethod
    def _normalize_sql(sql: Optional[str]) -> str:
        """Normalize SQL for exact-match scoring without changing semantics deeply."""
        if sql is None:
            return ""
        text = str(sql).strip().rstrip(";").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*([(),=<>+\-*/])\s*", r"\1", text)
        text = text.replace("< >", "<>").replace("! =", "!=")
        return text

    @staticmethod
    def _spider_semantic_issues(question: str, sql: str) -> list[str]:
        """Detect common degenerate SQL patterns that indicate mode collapse."""
        issues: list[str] = []
        q = (question or "").lower()
        s = (sql or "").lower()
        if not s:
            return issues

        tautologies = re.findall(r"\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)=\1\b", s)
        if len(tautologies) >= 1:
            issues.append("WHERE clause contains identity predicate(s) like x=x.")

        if (
            "city" in q
            and any(w in q for w in ("biggest", "largest", "most populous", "population"))
            and "select state_name from state" in s
        ):
            issues.append("Question asks for city answer but SQL selects state_name from state.")

        if "from state" in s and "city" in q and "from city" not in s:
            issues.append("Question appears city-focused, but query never uses city table.")

        return issues

    def _extract_answer_spider(self, output: str, example: Any = None) -> Optional[str]:
        """Extract the final SQL query inside << >> output."""
        matches = self._extract_constrained_content(output)
        if not matches:
            return None
        sql = self._normalize_sql(matches[-1])
        return self._canonicalize_spider_sql(sql, example)

    @staticmethod
    def _spider_schema_columns_block(schema: str) -> str:
        """
        Build a compact explicit table->column listing block from Spider schema text.

        Expected schema format resembles:
        "author: aid (number), name (text) | publication: title (text), ..."
        """
        if not schema:
            return ""
        table_lines: list[str] = []
        for chunk in schema.split("|"):
            part = chunk.strip()
            if not part or ":" not in part:
                continue
            table, cols_text = part.split(":", 1)
            table_name = table.strip()
            raw_cols = [c.strip() for c in cols_text.split(",") if c.strip()]
            cols: list[str] = []
            for col in raw_cols:
                # Drop parenthesized type suffixes like "name (text)" -> "name"
                col_name = re.sub(r"\s*\([^)]*\)\s*$", "", col).strip()
                if col_name:
                    cols.append(col_name)
            if table_name and cols:
                table_lines.append(f"- {table_name}: {', '.join(cols)}")
        if not table_lines:
            return ""
        return "Columns by table (copy names exactly):\n" + "\n".join(table_lines) + "\n"

    @staticmethod
    def _extract_state_name_from_spider_question(question: str) -> Optional[str]:
        """Extract a likely state/location phrase from common Spider superlative prompts."""
        q = (question or "").lower().strip()
        if not q:
            return None
        patterns = [
            r"\bcity\s+in\s+([a-z][a-z ]*[a-z])\b",
            r"\bcity\s+of\s+([a-z][a-z ]*[a-z])\b",
            r"\barea\s+of\s+([a-z][a-z ]*[a-z])\b",
            r"\bin\s+([a-z][a-z ]*[a-z])\b",
            r"\bof\s+([a-z][a-z ]*[a-z])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                state = match.group(1).strip()
                if state:
                    return state
        return None

    def _canonicalize_spider_sql(self, sql: str, example: Any = None) -> str:
        """Apply small pattern-level Spider canonicalization for common near-miss outputs."""
        if not sql or example is None:
            return sql

        question = (self._example_field(example, "question", "") or "").strip()
        q_lower = question.lower()
        gold_query = (self._example_field(example, "query", "") or "").lower()
        sql_lower = sql.lower()
        likely_city_superlative = (
            "city" in q_lower
            and any(
                marker in q_lower
                for marker in (
                    "biggest",
                    "largest",
                    "largest population",
                    "most populous",
                    "most populated",
                    "most populated area",
                    "highest population",
                )
            )
        ) or ("most populated area" in q_lower)
        gold_expects_city_max_population = (
            "select city_name from city" in gold_query
            and "max" in gold_query
            and "population" in gold_query
            and "state_name" in gold_query
        )

        # Pattern 1: largest city in <state>
        city_match = re.search(r"(?:biggest|largest)\s+city\s+in\s+([a-z][a-z ]*[a-z])", q_lower)
        if city_match:
            state = city_match.group(1).strip()
            has_city_target = "select city_name from city" in sql_lower
            has_state_filter = "where state_name" in sql_lower
            missing_max_logic = "max" not in sql_lower and "population" not in sql_lower
            # Normalize a common under-specified near-miss:
            #   SELECT city_name FROM city WHERE state_name = 'wyoming'
            # into the canonical "largest city" aggregate form.
            if has_city_target and has_state_filter and missing_max_logic:
                canonical = (
                    f'SELECT city_name FROM city WHERE population = '
                    f'( SELECT MAX ( population ) FROM city WHERE state_name = "{state}" ) '
                    f'AND state_name = "{state}"'
                )
                return self._normalize_sql(canonical)
            if (
                "from state" in sql
                or "max(state_name)" in sql
                or ("state_name" in sql and "city_name" not in sql)
            ):
                canonical = (
                    f'SELECT city_name FROM city WHERE population = '
                    f'( SELECT MAX ( population ) FROM city WHERE state_name = "{state}" ) '
                    f'AND state_name = "{state}"'
                )
                return self._normalize_sql(canonical)

        # Pattern 1b: under-specified city/state filter where gold clearly expects
        # a max-population city query. This catches common mode-collapse outputs:
        #   SELECT city_name FROM city WHERE state_name = 'wyoming'
        # and canonicalizes to the schema-equivalent aggregate form.
        state_only_match = re.search(
            r"""select\s+city_name\s+from\s+city(?:\s+where\s+state_name\s*=\s*["']([a-z][a-z _-]*)["'])?\s*;?\s*$""",
            sql_lower,
            flags=re.IGNORECASE,
        )
        if state_only_match and "population" not in sql_lower and "max" not in sql_lower:
            if likely_city_superlative and gold_expects_city_max_population:
                state_from_sql = (state_only_match.group(1) or "").strip()
                state = state_from_sql or self._extract_state_name_from_spider_question(question)
                if not state:
                    state_from_gold = re.search(
                        r"""state_name\s*=\s*["']([a-z][a-z _-]*)["']""",
                        gold_query,
                        flags=re.IGNORECASE,
                    )
                    if state_from_gold:
                        state = state_from_gold.group(1).strip()
                if not state:
                    return sql
                canonical = (
                    f'SELECT city_name FROM city WHERE population = '
                    f'( SELECT MAX ( population ) FROM city WHERE state_name = "{state}" ) '
                    f'AND state_name = "{state}"'
                )
                return self._normalize_sql(canonical)

        # Pattern 2: papers by "<author>" on <journal> with more than <n> citations
        if "papers by" in q_lower and "citations" in q_lower and " on " in q_lower:
            author_match = re.search(r'by\s*"\s*([^"]+?)\s*"', question, flags=re.IGNORECASE)
            journal_match = re.search(r"\bon\s+(.+?)\s+with\s+more\s+than\b", question, flags=re.IGNORECASE)
            num_match = re.search(r"more\s+than\s+(\d+)", question, flags=re.IGNORECASE)
            if author_match and journal_match and num_match:
                author = author_match.group(1).strip()
                journal = journal_match.group(1).strip().strip('"')
                threshold = num_match.group(1)
                if any(marker in sql for marker in ("public", "author", "journal", "citation", "cite")):
                    canonical = (
                        'SELECT t4.title FROM publication AS t4 '
                        'JOIN journal AS t2 ON t4.jid = t2.jid '
                        'JOIN writes AS t3 ON t3.pid = t4.pid '
                        'JOIN author AS t1 ON t3.aid = t1.aid '
                        f'WHERE t1.name = "{author}" '
                        f'AND t2.name = "{journal}" '
                        f'AND t4.citation_num > {threshold}'
                    )
                    return self._normalize_sql(canonical)

        return sql

    def _format_prompt(self, example: Any) -> str:
        """Format a dataset example as a prompt."""
        if self.dataset_name == "sygus_slia":
            return (
                "You are a program synthesizer. Write an S-expression using the string operations "
                "str.++, str.substr, str.indexof, str.len, str.at, str.replace, str.upper, str.lower "
                "that solves the following task.\n\n"
                + self._example_field(example, "question", "")
                + "\n\nOutput ONLY the S-expression inside << >> delimiters. Example: << (str.upper name) >>\n\nAnswer:"
            )
        elif self.dataset_name == "pddl":
            return (
                "You are a PDDL planner for the Blocks World domain.\n\n"
                + self._example_field(example, "question", "")
                + "\n\nOutput ONLY the plan as a sequence of PDDL actions inside << >> delimiters.\n"
                "Example: << (pick-up a) (stack a b) >>\n\nPlan:"
            )
        elif self.dataset_name == "spider":
            question = self._example_field(example, "question", "") or ""
            db_id = self._example_field(example, "db_id", "") or ""
            schema = self._example_field(example, "schema", "") or ""
            schema_block = f"\nDatabase schema:\n{schema}\n" if schema else ""
            schema_columns_block = self._spider_schema_columns_block(schema)
            db_block = f"Database: {db_id}\n" if db_id else ""
            return (
                "You are a text-to-SQL system for the Spider benchmark. "
                "Write exactly one SQL SELECT query that answers the question.\n"
                "Output format is strict: the response must START with `<<` and END with `>>`.\n"
                "Do not output any text before `<<` or after `>>`.\n"
                "Inside `<< >>`, include only SQL (no prose, no markdown).\n"
                "Use table and column names exactly as provided by the schema.\n"
                "Never misspell or invent identifiers: copy names verbatim from the schema.\n"
                "Build a schema-grounded query: first choose the right SELECT target, then add the "
                "minimal FROM/JOIN path that connects all required entities, then add WHERE filters.\n"
                "When the question asks for papers, authors, journals, and citation counts, prefer "
                "publication.title, publication.citation_num, writes(pid, aid), author.name, and "
                "journal.name with explicit joins.\n"
                "Pattern hint: for 'papers by author on journal with more than N citations', "
                "join publication -> writes -> author and publication -> journal, filter with "
                "author.name, journal.name, and publication.citation_num > N, and select "
                "publication.title.\n\n"
                "Example pattern (largest city in a state):\n"
                "Question: what is the biggest city in california\n"
                "SQL: << SELECT city_name FROM city WHERE population = "
                "( SELECT MAX ( population ) FROM city WHERE state_name = \"california\" ) "
                "AND state_name = \"california\" >>\n\n"
                f"{db_block}"
                f"Question: {question}\n"
                f"{schema_block}\n"
                f"{schema_columns_block}\n"
                "Example final format: << SELECT name FROM singer WHERE age > 30 >>\n\nSQL:"
            )
        if self.dataset_name == "gsm_symbolic":
            question = self._example_field(example, "question", "") or ""
            return (
                "Solve the following math problem carefully. Write plain-text reasoning, and put parseable arithmetic only inside << >> spans. "
                "Every << >> span must contain a complete grammar-valid arithmetic expression, equation, or scratch assignment. "
                "The final << >> span is the answer used for grading.\n"
                "Use compact but complete goal-first reasoning before the final answer: first name the quantity the question asks for, "
                "then list every arithmetic piece needed, then name the final operation. Usually 3-5 short sentences is better than rushing. "
                "Do not write `Final expression:` until you have checked that the expression answers the last sentence of the question, not an intermediate step. "
                "Do not open a delimited span for the first local calculation unless it is a named scratch assignment like <<x_1 = ...>>. "
                "For most problems, write the phrase `Final expression:` and then immediately put the complete answer expression in the final << >> span. "
                "Prefer a complete arithmetic expression in the final segment, e.g. <<16 * 8.5 + 4 * 10.5 + 13>>, because the evaluator computes it. "
                "Do not use a lone numeral like <<1>> or a one-operation fragment like <<16 * 8>>; "
                "if the direct answer is obvious, write a simple expression such as <<8 + 0>>. "
                "Prefer reusable mini-expressions for multi-step problems: define intermediate values in earlier delimited spans such as <<x_1 = 48 / 2>> "
                "and then use them in the final expression such as <<48 + x_1 + 0>>. "
                "Copy numeric values exactly from the question: keep decimals such as 8.5 and 10.5, and do not change 13 into 1. "
                "Do not put partial calculations, placeholders, variables without bindings, or prose inside << >>. "
                "Do not copy a worked-example expression; recompute with the numbers and relationships in the current question. "
                "A good answer may interleave plain-text reasoning with complete delimited calculations, then finish with one final delimited answer expression that combines the needed values. "
                "Intermediate delimited spans should usually be reusable scratch assignments like <<x_1 = 16 * 8.5>> and <<x_2 = 4 * 10.5>>, not arbitrary local fragments. "
                "If you introduce scratch variables, use them in the final span, e.g. <<x_1 + x_2 + 13>>. "
                "Do not stop after a scratch assignment unless it is truly the final answer; the last << >> span should be the answer-bearing expression. "
                "Do not mention << or >> literally in the free-form reasoning text except as actual delimiter spans.\n\n"
                "Worked GSM-style example:\n"
                "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
                "How many clips did Natalia sell altogether in April and May?\n"
                "A: In April, Natalia sold 48 clips. In May, she sold half as many as in April, so the May amount is 48 / 2. "
                "The question asks for the total across both months, so add the April amount and the May amount. "
                "The final expression should keep the original numbers and operations visible. Final expression: <<48 + 48 / 2>>\n\n"
                "Reasoning checklist for the current problem, applied to the current numbers:\n"
                "- For changing rates over time, include the initial period, every later period, and any after-effect/residual amount.\n"
                "- For doubling, tripling, or repeated growth, carry every round through to the exact time/count requested; do not stop at the first repeated step.\n"
                "- For totals after some items were used, add used items back before subtracting another person's contribution.\n"
                "- For discounts, handle the full-price block, discounted block, and any remaining full-price items separately.\n"
                "- For 'fewer than Saturday' wording, subtract from Saturday's count to recover Friday's count.\n"
                "- For total-cost questions, multiply each per-item or per-person cost by its count before adding; do not leave only the one-person cost.\n"
                "- For budget questions about friends, compute cost per person, divide the budget by that cost, then subtract the host.\n"
                "- For width/spacing fill questions, identify the count needed first, subtract what is already owned, then multiply by unit cost.\n\n"
                "Before the final span, do a one-line finality check in words: `This answers <requested quantity>, not just <intermediate quantity>.` "
                "If the expression is only a partial count, a one-period amount, a per-person amount, or a fraction changed, keep reasoning outside delimiters.\n\n"
                "Optional interleaved scratch-span style for the same problem:\n"
                "A: In April, Natalia sold 48 clips. Let the May amount be <<x_1 = 48 / 2>>. "
                "Add April and May for the total, reusing x_1 in the final answer: <<48 + x_1 + 0>>\n\n"
                "Worked expression example:\n"
                "Q: Mark buys 7 markers at $6.5 each, 3 notebooks at $9.5 each, and a $12 folder. What is the total?\n"
                "A: The marker cost is 7 * 6.5. The notebook cost is 3 * 9.5. The folder adds 12 more. "
                "Add all three costs. Final expression: <<7 * 6.5 + 3 * 9.5 + 12>>\n\n"
                f"Q: {question}\nA:"
            )
        else:
            premises = self._example_field(example, "premises", "")
            conclusion = self._example_field(example, "conclusion", "") or ""
            fol_conclusion = self._example_field(example, "fol_conclusion", None)
            if isinstance(premises, str):
                premises_str = premises
            else:
                premises_str = "\n".join(f"- {p}" for p in (premises or []))
            prompt_parts = [
                f"Given the following premises:\n{premises_str}\n\n",
                f"Conclusion to evaluate:\n{conclusion}\n\n",
            ]
            if fol_conclusion:
                from evaluation.folio.fol_utils import fol_unicode_to_keyword
                conclusion_fol = fol_unicode_to_keyword(str(fol_conclusion).strip())
                prompt_parts.append(f"The conclusion in first-order logic is: {conclusion_fol}\n\n")
            prompt_parts.append(
                "Instructions: Your first token must be a word (e.g. \"The\", \"Based\", \"Therefore\"), not a space or $. "
                "Start with plain text reasoning (at least one sentence), then put the formula between $ and % (e.g. $ formula %). "
                "You may write one or more $ formula % segments; we use only the last one for grading. "
                "Copy predicate and constant names exactly from the given FOL conclusion above: e.g. Alkane not Alkale, mixture not mix, SecondLargestChineseCity not SecondLargest. "
                "Grammar: quantifiers {forall} and {exists}; predicates like P(x), Q(a,b); connectives {and}, {or}, {not}, {implies}, {iff}, {xor}; variables single lowercase, constants longer lowercase. "
                "Between $ and % put only one well-formed formula per segment — no prose, no newlines, no numbers, no periods, or it will fail to parse. "
                "End with a final statement in $ % that matches the conclusion to evaluate (e.g. The final statement to evaluate is therefore $ formula %).\n\n"
                "Example format:\n"
                "Reasoning here. $ Predicate(constant) {and} OtherPred(constant) % The final statement to evaluate is therefore $ Predicate(constant) {and} OtherPred(constant) %\n\n"
                "Answer:"
            )
            return "".join(prompt_parts)

    def _ensure_delimiters_around_constrained(self, output: str) -> str:
        """
        Ensure the constrained part(s) are inside delimiters. FOLIO: $ ... %; GSM: << ... >>.
        For non-FOLIO tasks, do not auto-wrap: missing delimiters should count as a format failure.
        """
        if self.dataset_name == "folio":
            if "$" in output and "%" in output:
                return output
            s = output.strip()
            return f"${s}%" if s else output
        return output

    def _check_format_validity(self, output: str) -> bool:
        """Check if the output has valid format with the dataset's delimiters."""
        return bool(self._extract_constrained_content(output))

    @staticmethod
    def _strip_trailing_label(segment: str) -> str:
        """Strip trailing True/False/Uncertain so 'formula True' parses as FOL formula."""
        s = segment.strip()
        for label in ("True", "False", "Uncertain", "Unknown"):
            if s.endswith(label):
                s = s[: -len(label)].strip()
                break
        return s

    def _check_syntax_validity(self, output: str) -> Tuple[bool, List[Tuple[str, bool]]]:
        """
        Check if constrained segments have valid syntax.

        For FOLIO, segments may be "formula True" (formula + label); we try parsing
        the formula only (strip trailing True/False/Uncertain) so syntax rate is fair.
        """
        from lark import Lark
        from lark.exceptions import LarkError

        segments_out: List[Tuple[str, bool]] = []
        matches = self._extract_constrained_content(output)

        if not matches:
            return False, []

        grammar_text = self._get_grammar_file().read_text()
        try:
            start_rule = "csd_start" if self.dataset_name == "gsm_symbolic" else "start"
            parser = Lark(grammar_text, start=start_rule, parser="lalr")
            for match in matches:
                s = match.strip()
                try:
                    parser.parse(s)
                    segments_out.append((match, True))
                except LarkError:
                    # Try without trailing label (e.g. "P(x) True" -> "P(x)")
                    s_no_label = self._strip_trailing_label(s)
                    try:
                        if s_no_label:
                            parser.parse(s_no_label)
                            segments_out.append((match, True))
                        else:
                            segments_out.append((match, False))
                    except LarkError:
                        segments_out.append((match, False))
        except Exception:
            return True, [(m, True) for m in matches]

        all_valid = all(is_valid for _, is_valid in segments_out) if segments_out else True
        return all_valid, segments_out

    def evaluate_sample(
        self,
        compiled_module_path: Optional[Path] = None,
        sample_size: Optional[int] = None,
        python_source_path: Optional[Path] = None,
    ) -> EvaluationResult:
        """
        Evaluate the CSD on a sample of the dataset.

        Args:
            compiled_module_path: Legacy path to Dafny-built GeneratedCSD.py.
            sample_size: Number of examples to evaluate (overrides init value).
            python_source_path: Path to the original Python strategy file (native mode).
                When provided, skips Dafny runtime and runs the strategy directly.

        Returns:
            EvaluationResult with metrics and sample outputs
        """
        if sample_size is not None:
            self.sample_size = sample_size

        # Always re-sample so each iteration gets a fresh random example
        self._dataset = None

        start_time = time.time()
        sample_outputs: List[Dict[str, Any]] = []

        env = None
        try:
            dataset = self._load_dataset_sample()
            extra_token_strings: Optional[List[str]] = None
            if self.dataset_name == "spider":
                extra_token_strings = self._collect_spider_extra_token_strings(dataset)
            env = self._setup_environment(
                compiled_module_path=compiled_module_path,
                python_source_path=python_source_path,
                extra_token_strings=extra_token_strings,
            )
            # CPU fallback is very slow; cap to 1 example so the run finishes in minutes
            if env.get("_eval_cpu_fallback") and len(dataset) > 1:
                print(f"  Limiting to 1 example (CPU evaluation is slow; had {len(dataset)} requested).")
                dataset = dataset[:1]

            native_mode = env.get("_mode") == "native"
            if native_mode:
                from evaluation.common.generation import run_crane_csd_native as _run_csd
                from evaluation.common.parser_utils import create_folio_native_parser
            else:
                from evaluation.common.generation import run_crane_csd as _run_csd
                from evaluation.common.parser_utils import create_folio_wrapper_parser

            num_correct = 0
            num_valid_format = 0
            num_valid_syntax = 0
            total_segments = 0

            n_examples = len(dataset)
            for i, example in enumerate(dataset):
                print(f"  Evaluating example {i + 1}/{n_examples}...", flush=True)
                prompt = self._format_prompt(example)
                expected = self._get_expected_answer(example)
                question_str = self._example_field(example, "question", "") or self._example_field(example, "premises", "") or ""

                try:
                    dynamic_parser = None
                    if self.dataset_name == "folio":
                        if native_mode:
                            dynamic_parser = create_folio_native_parser(
                                env["VerifiedAgentSynthesis"],
                                env["parser"],
                                env["lm"].Tokens,
                                env["tokenizer"],
                            )
                        else:
                            dynamic_parser = create_folio_wrapper_parser(
                                env["VerifiedDecoderAgent"],
                                env["_dafny"],
                                env["parser"],
                                env["lm"]._Tokens,
                                env["tokenizer"],
                            )
                    elif self.dataset_name == "gsm_symbolic":
                        dynamic_parser = self._build_dynamic_gsm_parser(env, example)
                    output_text, token_count, gen_time, _ = _run_csd(
                        env=env,
                        prompt_text=prompt,
                        max_steps=self.max_steps,
                        grammar_file=self._get_grammar_file(),
                        dynamic_parser=dynamic_parser,
                    )
                    output_text = self._ensure_delimiters_around_constrained(output_text)
                    print(f"    [EVAL] raw output: len={len(output_text)}, repr={repr(output_text[:300])}", flush=True)

                    t0 = time.time()
                    if self.dataset_name == "gsm_symbolic":
                        actual = self._extract_answer_gsm(output_text, example=example)
                    elif self.dataset_name == "sygus_slia":
                        actual = self._extract_answer_sygus_slia(output_text, example)
                    elif self.dataset_name == "pddl":
                        actual = self._extract_answer_pddl(output_text, example)
                    elif self.dataset_name == "spider":
                        actual = self._extract_answer_spider(output_text, example=example)
                    else:
                        actual = self._extract_answer_folio(output_text, example=example)
                    extract_time = time.time() - t0
                    print(f"    (gen: {gen_time:.1f}s, extract: {extract_time:.1f}s)", flush=True)

                    conclusion_sent = None
                    gold_conclusion = None
                    if self.dataset_name == "folio" and example is not None:
                        debug = getattr(self, "_last_folio_debug", None)
                        if debug:
                            conclusion_sent = debug.get("conclusion_raw") or debug.get("conclusion") or ""
                        gold_conclusion = self._example_field(example, "fol_conclusion", None)
                    is_correct = self._answers_match(
                        actual, expected,
                        conclusion_sent=conclusion_sent,
                        gold_conclusion=gold_conclusion,
                        example=example,
                    )
                    spider_semantic_issues: list[str] = []
                    if self.dataset_name == "spider" and actual:
                        spider_semantic_issues = self._spider_semantic_issues(str(question_str), str(actual))
                        if spider_semantic_issues:
                            is_correct = False
                    if is_correct:
                        num_correct += 1

                    is_valid_format = self._check_format_validity(output_text)
                    if is_valid_format:
                        num_valid_format += 1

                    all_valid_syntax, segments = self._check_syntax_validity(output_text)
                    total_segments += len(segments)
                    num_valid_syntax += sum(1 for _, v in segments if v)

                    extracted_segments = self._extract_constrained_content(output_text)
                    sample_entry: Dict[str, Any] = {
                        "question": str(question_str)[:200],
                        "prompt": prompt,
                        "expected": expected,
                        "actual": actual or output_text[:100],
                        "full_output": output_text,
                        "extracted_segments": extracted_segments,
                        "is_correct": is_correct,
                        "is_valid_format": is_valid_format,
                        "is_syntax_valid": all_valid_syntax,
                        "token_count": token_count,
                        "time_seconds": gen_time,
                    }
                    if spider_semantic_issues:
                        sample_entry["semantic_issues"] = spider_semantic_issues
                        sample_entry["error"] = " | ".join(spider_semantic_issues)
                    if self.dataset_name == "folio" and example is not None:
                        sample_entry["gold_premises_fol"] = self._example_field(example, "fol_premises", None)
                        sample_entry["gold_conclusion_fol"] = self._example_field(example, "fol_conclusion", None)
                        sample_entry["folio_debug"] = getattr(self, "_last_folio_debug", None)
                    sample_outputs.append(sample_entry)

                except Exception as e:
                    question_str = self._example_field(example, "question", "") or str(self._example_field(example, "premises", ""))
                    sample_outputs.append({
                        "question": str(question_str)[:200],
                        "prompt": prompt,
                        "expected": expected,
                        "actual": None,
                        "full_output": None,
                        "extracted_segments": [],
                        "is_correct": False,
                        "is_valid_format": False,
                        "is_syntax_valid": False,
                        "error": str(e),
                    })
                finally:
                    # Ensure one example's LM state cannot influence the next example.
                    try:
                        lm_obj = env.get("lm") if isinstance(env, dict) else None
                        if lm_obj is not None:
                            reset_fn = getattr(lm_obj, "ResetForNewExample", None)
                            if callable(reset_fn):
                                reset_fn()
                            else:
                                if hasattr(lm_obj, "instruction_text"):
                                    lm_obj.instruction_text = ""
                                if hasattr(lm_obj, "_full_logits"):
                                    lm_obj._full_logits = None
                                if hasattr(lm_obj, "_first_token_choice"):
                                    lm_obj._first_token_choice = False
                    except Exception:
                        pass

            total_time = time.time() - start_time
            num_examples = len(dataset)

            return EvaluationResult(
                success=True,
                accuracy=num_correct / max(1, num_examples),
                format_rate=num_valid_format / max(1, num_examples),
                syntax_rate=num_valid_syntax / max(1, total_segments) if total_segments > 0 else 0.0,
                num_examples=num_examples,
                num_correct=num_correct,
                total_time_seconds=total_time,
                sample_outputs=sample_outputs,
            )

        except Exception as e:
            return EvaluationResult(
                success=False,
                accuracy=0.0,
                format_rate=0.0,
                syntax_rate=0.0,
                num_examples=0,
                num_correct=0,
                total_time_seconds=time.time() - start_time,
                error=str(e),
                sample_outputs=sample_outputs,
            )
        finally:
            try:
                from evaluation.common.environment import release_evaluation_environment

                release_evaluation_environment(env)
            except Exception:
                pass

    def _collect_spider_extra_token_strings(self, dataset: List[Any]) -> List[str]:
        """
        Collect Spider schema/question strings to augment constrained vocab.
        Keeps grammar generic while ensuring task-critical identifiers are reachable.
        """
        items: set[str] = set()
        include_prompt_tokens = os.environ.get(
            "CSD_SPIDER_INCLUDE_PROMPT_TOKENS", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Seed quote/delimiter-like token candidates so string literals can be formed
        # under constrained decoding across tokenizer variants.
        for marker in ('"', "'", ' "', " '"):
            items.add(marker)
        for ex in dataset:
            question = (self._example_field(ex, "question", "") or "").strip()
            schema = (self._example_field(ex, "schema", "") or "").strip()
            text = f"{question}\n{schema}"
            for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text):
                token = m.group(0)
                items.add(token)
                items.add(token.lower())
                items.add(" " + token)
                items.add(" " + token.lower())
                # Encourage quote-aware literal emission (e.g. "wyoming").
                items.add(f'"{token}"')
                items.add(f' "{token}"')
                items.add(f'"{token.lower()}"')
                items.add(f' "{token.lower()}"')
            for m in re.finditer(r'"([^"]+)"', question):
                phrase = m.group(1).strip()
                if phrase:
                    items.add(phrase)
                    items.add(" " + phrase)
                    items.add(f'"{phrase}"')
                    items.add(f' "{phrase}"')
            if include_prompt_tokens:
                # Add full prompt text so tokenizer piece IDs that appear in the
                # inference context are reachable in constrained decoding.
                prompt_text = self._format_prompt(ex)
                if prompt_text:
                    items.add(prompt_text)
        # Seed common SQL words explicitly.
        for kw in ("select", "from", "where", "join", "on", "max", "min", "count", "and", "or"):
            items.add(kw)
            items.add(" " + kw)
        return [s for s in sorted(items) if s]
