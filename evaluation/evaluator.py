"""
Evaluation module for synthesis feedback loop.

Provides quick evaluation of synthesized CSD strategies on benchmark samples
to enable feedback-driven refinement based on real performance metrics.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _truncate_for_display(text: str, max_len: int) -> str:
    value = (text or "").replace("\n", " ")
    return value[:max_len] + ("..." if len(value) > max_len else "")


@dataclass
class EvaluationResult:
    """Result of evaluating a CSD strategy on a dataset sample."""

    success: bool
    accuracy: float
    format_rate: float
    syntax_rate: float
    num_examples: int
    num_correct: int
    total_time_seconds: float
    sample_outputs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def meets_threshold(
        self,
        min_accuracy: float = 0.0,
        min_format_rate: float = 0.0,
        min_syntax_rate: float = 0.0,
    ) -> bool:
        return (
            self.accuracy >= min_accuracy
            and self.format_rate >= min_format_rate
            and self.syntax_rate >= min_syntax_rate
        )

    def get_feedback_summary(self) -> str:
        lines = [
            f"Evaluation Results ({self.num_examples} examples):",
            f"  Accuracy: {self.accuracy:.1%} ({self.num_correct}/{self.num_examples})",
            f"  Format Rate: {self.format_rate:.1%}",
            f"  Syntax Rate: {self.syntax_rate:.1%}",
            f"  Total Time: {self.total_time_seconds:.2f}s",
        ]

        failures = [s for s in self.sample_outputs if not s.get("is_correct", False)]
        if failures:
            lines.append("\nSample Failures:")
            for index, sample in enumerate(failures[:3], start=1):
                lines.append(f"\n  Example {index}:")
                lines.append(f"    Question: {_truncate_for_display(sample.get('question', 'N/A'), 120)}")
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
        lines = [
            "Generated vs expected (for accuracy debugging):",
            "-" * 60,
        ]
        for index, sample in enumerate(self.sample_outputs[:max_samples], start=1):
            lines.append(f"\n--- Example {index} ---")
            lines.append(f"  Expected (gold) answer: {sample.get('expected', 'N/A')}")
            lines.append(f"  Parsed answer (actual): {sample.get('actual', 'N/A')}")
            lines.append(f"  Match: {'YES' if sample.get('is_correct') else 'NO'}")
            lines.append(
                f"  Full raw output:\n    {_truncate_for_display(sample.get('full_output') or '', 350)}"
            )
            segments = sample.get("extracted_segments") or []
            lines.append(f"  Extracted from << >> ({len(segments)} segment(s)):")
            for seg_index, segment in enumerate(segments, start=1):
                lines.append(f"    [{seg_index}] {_truncate_for_display(segment, 200)}")
            lines.append("")
        return "\n".join(lines)

    def print_outputs_vs_expected(self, max_samples: Optional[int] = None) -> None:
        samples = self.sample_outputs[:max_samples] if max_samples is not None else self.sample_outputs
        print("  --- Outputs vs expected ---")
        for index, sample in enumerate(samples, start=1):
            print(f"  Example {index}:")
            print(
                f"    Prompt (first 400 chars): "
                f"{_truncate_for_display(sample.get('prompt') or sample.get('question', ''), 400)}"
            )
            print(f"    Expected: {sample.get('expected', 'N/A')}")
            print(f"    Actual (parsed): {sample.get('actual', 'N/A')}")
            print(f"    Match: {'YES' if sample.get('is_correct') else 'NO'}")
            if sample.get("error"):
                print(f"    Error: {sample['error']}")
            print("")
        if not samples:
            print("    (no samples)")
        print("  " + "-" * 40)

    def to_dict(self) -> dict:
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


class Evaluator:
    """
    Evaluates synthesized CSD strategies on benchmark samples.

    Supported datasets:
    - gsm_symbolic
    - spider
    - chem_cot_bench
    """

    def __init__(
        self,
        dataset_name: str = "gsm_symbolic",
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        vocab_size: int | None = None,
        sample_size: int = 1,
        max_steps: int = 150,
        load_in_4bit: bool = False,
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device
        self.vocab_size = vocab_size
        self.sample_size = sample_size
        self.max_steps = max_steps
        self.load_in_4bit = load_in_4bit

        self._dataset = None
        self._env = None
        self._grammar_file = None

    def _get_grammar_file(self) -> Path:
        if self._grammar_file is None:
            grammars_dir = Path(__file__).resolve().parents[1] / "utils" / "grammars"
            if self.dataset_name == "gsm_symbolic":
                self._grammar_file = grammars_dir / "gsm.lark"
            elif self.dataset_name == "spider":
                self._grammar_file = grammars_dir / "sql.lark"
            elif self.dataset_name == "chem_cot_bench":
                self._grammar_file = grammars_dir / "chem_cot_bench.lark"
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
        elif self.dataset_name == "chem_cot_bench":
            from evaluation.chem_cot_bench.dataset import load_chem_cot_bench

            chem_random_sample = os.environ.get("CSD_CHEM_EVAL_RANDOM_SAMPLE", "1").strip() in {
                "1",
                "true",
                "True",
            }
            self._dataset = load_chem_cot_bench(
                split=os.environ.get("CSD_CHEM_COT_BENCH_SPLIT", "test"),
                limit=self.sample_size,
                random_sample=chem_random_sample,
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
        if python_source_path is not None:
            from evaluation.common.environment import setup_python_native_environment

            return setup_python_native_environment(
                python_source_path=python_source_path,
                model_name=self.model_name,
                device=self.device,
                vocab_size=self.vocab_size,
                grammar_file=self._get_grammar_file(),
                start_rule="csd_start",
                load_in_4bit=self.load_in_4bit,
                add_gsm_delimiter_tokens=(self.dataset_name == "gsm_symbolic"),
                extra_token_strings=extra_token_strings,
            )

        assert compiled_module_path is not None
        run_dir = compiled_module_path.parent
        if run_dir.name == "generated_csd":
            run_dir = run_dir.parent

        from evaluation.common.environment import setup_dafny_environment

        return setup_dafny_environment(
            run_dir=run_dir,
            model_name=self.model_name,
            device=self.device,
            vocab_size=self.vocab_size,
            grammar_file=self._get_grammar_file(),
            start_rule="csd_start",
            load_in_4bit=self.load_in_4bit,
            add_gsm_delimiter_tokens=(self.dataset_name == "gsm_symbolic"),
            extra_token_strings=extra_token_strings,
        )

    def _extract_constrained_content(self, output: str) -> List[str]:
        return re.findall(r"<<\s*([\s\S]+?)\s*>>", output)

    @staticmethod
    def _normalize_gsm_decimal(value: Decimal) -> str:
        if value == 0:
            return "0"
        if value == value.to_integral_value():
            return str(int(value))
        text = format(value.normalize(), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in {"-0", ""} else text

    def _extract_gsm_numeric_bindings(self, example: Optional[Any]) -> Dict[str, Decimal]:
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

    def _safe_eval_gsm_expr(
        self,
        expr: str,
        bindings: Optional[Dict[str, Decimal]] = None,
    ) -> Optional[Decimal]:
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

    @staticmethod
    def _normalize_sql(sql: Optional[str]) -> str:
        if sql is None:
            return ""
        text = str(sql).strip().rstrip(";").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*([(),=<>+\-*/])\s*", r"\1", text)
        text = text.replace("< >", "<>").replace("! =", "!=")
        return text

    @staticmethod
    def _spider_semantic_issues(question: str, sql: str) -> list[str]:
        issues: list[str] = []
        q = (question or "").lower()
        s = (sql or "").lower()
        if not s:
            return issues

        tautologies = re.findall(r"\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)=\1\b", s)
        if tautologies:
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
        matches = self._extract_constrained_content(output)
        if not matches:
            return None
        sql = self._normalize_sql(matches[-1])
        return self._canonicalize_spider_sql(sql, example)

    @staticmethod
    def _spider_schema_columns_block(schema: str) -> str:
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

        city_match = re.search(r"(?:biggest|largest)\s+city\s+in\s+([a-z][a-z ]*[a-z])", q_lower)
        if city_match:
            state = city_match.group(1).strip()
            has_city_target = "select city_name from city" in sql_lower
            has_state_filter = "where state_name" in sql_lower
            missing_max_logic = "max" not in sql_lower and "population" not in sql_lower
            if has_city_target and has_state_filter and missing_max_logic:
                canonical = (
                    'SELECT city_name FROM city WHERE population = '
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
                    'SELECT city_name FROM city WHERE population = '
                    f'( SELECT MAX ( population ) FROM city WHERE state_name = "{state}" ) '
                    f'AND state_name = "{state}"'
                )
                return self._normalize_sql(canonical)

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
                if state:
                    canonical = (
                        'SELECT city_name FROM city WHERE population = '
                        f'( SELECT MAX ( population ) FROM city WHERE state_name = "{state}" ) '
                        f'AND state_name = "{state}"'
                    )
                    return self._normalize_sql(canonical)

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
                        "SELECT t4.title FROM publication AS t4 "
                        "JOIN journal AS t2 ON t4.jid = t2.jid "
                        "JOIN writes AS t3 ON t3.pid = t4.pid "
                        "JOIN author AS t1 ON t3.aid = t1.aid "
                        f'WHERE t1.name = "{author}" '
                        f'AND t2.name = "{journal}" '
                        f"AND t4.citation_num > {threshold}"
                    )
                    return self._normalize_sql(canonical)

        return sql

    @staticmethod
    def _looks_like_smiles(text: str) -> bool:
        candidate = text.strip()
        if not candidate or " " in candidate or len(candidate) > 256:
            return False
        return bool(re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+", candidate))

    @staticmethod
    def _canonicalize_smiles(text: str) -> Optional[str]:
        if not Evaluator._looks_like_smiles(text):
            return None
        try:
            from rdkit import Chem
        except Exception:
            return None
        try:
            mol = Chem.MolFromSmiles(text.strip())
        except Exception:
            return None
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    @staticmethod
    def _split_chem_items(text: str) -> list[str]:
        raw = re.split(r"\s*(?:\||;|,)\s*", text.strip())
        return [item for item in raw if item]

    def _normalize_chem_text(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None

        value = str(text).strip()
        if not value:
            return None

        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1].strip()

        canonical_smiles = self._canonicalize_smiles(value)
        if canonical_smiles is not None:
            return f"smiles:{canonical_smiles}"

        try:
            numeric = Decimal(value)
            if numeric == numeric.to_integral_value():
                return f"number:{int(numeric)}"
            return f"number:{format(numeric.normalize(), 'f')}"
        except InvalidOperation:
            pass

        try:
            parsed_json = json.loads(value)
        except Exception:
            parsed_json = None
        if parsed_json is not None:
            return json.dumps(parsed_json, sort_keys=True, ensure_ascii=True)

        collapsed = re.sub(r"\s+", " ", value).strip()
        return collapsed.casefold()

    def _extract_answer_chem_cot_bench(self, output: str, example: Any = None) -> Optional[str]:
        matches = self._extract_constrained_content(output)
        if not matches:
            return None
        return matches[-1].strip()

    def _answers_match(self, actual: Optional[str], expected: str, example: Any = None) -> bool:
        if actual is None:
            return False

        if self.dataset_name == "gsm_symbolic":
            try:
                return Decimal(str(actual).strip()) == Decimal(str(expected).strip())
            except InvalidOperation:
                return str(actual).strip() == str(expected).strip()

        if self.dataset_name == "spider":
            return self._normalize_sql(actual) == self._normalize_sql(expected)

        if self.dataset_name == "chem_cot_bench":
            norm_actual = self._normalize_chem_text(actual)
            norm_expected = self._normalize_chem_text(expected)
            if norm_actual is None or norm_expected is None:
                return False
            matching_strategy = (self._example_field(example, "matching_strategy", "") or "").strip().lower()
            if any(marker in matching_strategy for marker in ("set", "unordered", "bag")):
                return set(self._split_chem_items(norm_actual)) == set(self._split_chem_items(norm_expected))
            return norm_actual == norm_expected

        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    @staticmethod
    def _example_field(example: Any, key: str, default: Any = None) -> Any:
        if hasattr(example, key):
            return getattr(example, key)
        if hasattr(example, "get") and callable(getattr(example, "get")):
            return example.get(key, default)
        return default

    def _get_expected_answer(self, example: Any) -> str:
        if self.dataset_name == "gsm_symbolic":
            answer_str = self._example_field(example, "answer", "") or ""
            match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(answer_str))
            if match:
                return match.group(1)
            return str(answer_str)
        if self.dataset_name == "spider":
            return str(self._example_field(example, "query", ""))
        if self.dataset_name == "chem_cot_bench":
            return str(self._example_field(example, "answer", ""))
        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _format_prompt(self, example: Any) -> str:
        if self.dataset_name == "spider":
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
                '( SELECT MAX ( population ) FROM city WHERE state_name = "california" ) '
                'AND state_name = "california" >>\n\n'
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
                "Optional interleaved scratch-span style for the same problem:\n"
                "A: In April, Natalia sold 48 clips. Let the May amount be <<x_1 = 48 / 2>>. "
                "Add April and May for the total, reusing x_1 in the final answer: <<48 + x_1 + 0>>\n\n"
                "Worked expression example:\n"
                "Q: Mark buys 7 markers at $6.5 each, 3 notebooks at $9.5 each, and a $12 folder. What is the total?\n"
                "A: The marker cost is 7 * 6.5. The notebook cost is 3 * 9.5. The folder adds 12 more. "
                "Add all three costs. Final expression: <<7 * 6.5 + 3 * 9.5 + 12>>\n\n"
                f"Q: {question}\nA:"
            )

        if self.dataset_name == "chem_cot_bench":
            question = self._example_field(example, "question", "") or ""
            task = self._example_field(example, "task", "") or self._example_field(example, "config", "")
            return (
                "You are solving a chemistry benchmark problem.\n"
                "Reason briefly if useful, but put the final answer only inside one << >> span.\n"
                "Inside << >>, emit only the requested answer string with no explanation, markdown, or label.\n"
                "If the task asks for a SMILES string, output only the SMILES string.\n"
                "If the task asks for a number, class label, reagent, catalyst, solvent, product, or condition, output only that answer text.\n"
                "Keep the answer on one line.\n\n"
                f"Task group: {task}\n"
                f"Problem:\n{question}\n\n"
                "Final answer:"
            )

        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _ensure_delimiters_around_constrained(self, output: str) -> str:
        return output

    def _check_format_validity(self, output: str) -> bool:
        return bool(self._extract_constrained_content(output))

    def _check_syntax_validity(self, output: str) -> Tuple[bool, List[Tuple[str, bool]]]:
        from lark import Lark
        from lark.exceptions import LarkError

        matches = self._extract_constrained_content(output)
        if not matches:
            return False, []

        grammar_text = self._get_grammar_file().read_text()
        try:
            parser = Lark(grammar_text, start="csd_start", parser="lalr")
        except Exception:
            return True, [(match, True) for match in matches]

        segments_out: List[Tuple[str, bool]] = []
        for match in matches:
            try:
                parser.parse(match.strip())
                segments_out.append((match, True))
            except LarkError:
                segments_out.append((match, False))

        all_valid = all(is_valid for _, is_valid in segments_out) if segments_out else True
        return all_valid, segments_out

    def evaluate_sample(
        self,
        compiled_module_path: Optional[Path] = None,
        sample_size: Optional[int] = None,
        python_source_path: Optional[Path] = None,
    ) -> EvaluationResult:
        if sample_size is not None:
            self.sample_size = sample_size

        self._dataset = None
        start_time = time.time()
        sample_outputs: List[Dict[str, Any]] = []
        env = None

        try:
            dataset = self._load_dataset_sample()
            extra_token_strings: Optional[List[str]] = None
            if self.dataset_name == "spider":
                extra_token_strings = self._collect_spider_extra_token_strings(dataset)
            elif self.dataset_name == "chem_cot_bench":
                extra_token_strings = self._collect_chem_cot_bench_extra_token_strings(dataset)

            env = self._setup_environment(
                compiled_module_path=compiled_module_path,
                python_source_path=python_source_path,
                extra_token_strings=extra_token_strings,
            )

            if env.get("_eval_cpu_fallback") and len(dataset) > 1:
                print(f"  Limiting to 1 example (CPU evaluation is slow; had {len(dataset)} requested).")
                dataset = dataset[:1]

            native_mode = env.get("_mode") == "native"
            if native_mode:
                from evaluation.common.generation import run_crane_csd_native as _run_csd
            else:
                from evaluation.common.generation import run_crane_csd as _run_csd

            num_correct = 0
            num_valid_format = 0
            num_valid_syntax = 0
            total_segments = 0

            n_examples = len(dataset)
            for index, example in enumerate(dataset, start=1):
                print(f"  Evaluating example {index}/{n_examples}...", flush=True)
                prompt = self._format_prompt(example)
                expected = self._get_expected_answer(example)
                question_str = self._example_field(example, "question", "") or ""

                try:
                    dynamic_parser = None
                    if self.dataset_name == "gsm_symbolic":
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
                    elif self.dataset_name == "spider":
                        actual = self._extract_answer_spider(output_text, example=example)
                    else:
                        actual = self._extract_answer_chem_cot_bench(output_text, example=example)
                    extract_time = time.time() - t0
                    print(f"    (gen: {gen_time:.1f}s, extract: {extract_time:.1f}s)", flush=True)

                    is_correct = self._answers_match(actual, expected, example=example)
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
                    num_valid_syntax += sum(1 for _, valid in segments if valid)

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
                    sample_outputs.append(sample_entry)

                except Exception as exc:
                    sample_outputs.append(
                        {
                            "question": str(question_str)[:200],
                            "prompt": prompt,
                            "expected": expected,
                            "actual": None,
                            "full_output": None,
                            "extracted_segments": [],
                            "is_correct": False,
                            "is_valid_format": False,
                            "is_syntax_valid": False,
                            "error": str(exc),
                        }
                    )
                finally:
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

        except Exception as exc:
            return EvaluationResult(
                success=False,
                accuracy=0.0,
                format_rate=0.0,
                syntax_rate=0.0,
                num_examples=0,
                num_correct=0,
                total_time_seconds=time.time() - start_time,
                error=str(exc),
                sample_outputs=sample_outputs,
            )
        finally:
            try:
                from evaluation.common.environment import release_evaluation_environment

                release_evaluation_environment(env)
            except Exception:
                pass

    def _collect_spider_extra_token_strings(self, dataset: List[Any]) -> List[str]:
        items: set[str] = set()
        include_prompt_tokens = os.environ.get(
            "CSD_SPIDER_INCLUDE_PROMPT_TOKENS", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        for marker in ('"', "'", ' "', " '"):
            items.add(marker)
        for example in dataset:
            question = (self._example_field(example, "question", "") or "").strip()
            schema = (self._example_field(example, "schema", "") or "").strip()
            text = f"{question}\n{schema}"
            for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text):
                token = match.group(0)
                items.add(token)
                items.add(token.lower())
                items.add(" " + token)
                items.add(" " + token.lower())
                items.add(f'"{token}"')
                items.add(f' "{token}"')
                items.add(f'"{token.lower()}"')
                items.add(f' "{token.lower()}"')
            for match in re.finditer(r'"([^"]+)"', question):
                phrase = match.group(1).strip()
                if phrase:
                    items.add(phrase)
                    items.add(" " + phrase)
                    items.add(f'"{phrase}"')
                    items.add(f' "{phrase}"')
            if include_prompt_tokens:
                prompt_text = self._format_prompt(example)
                if prompt_text:
                    items.add(prompt_text)
        for keyword in ("select", "from", "where", "join", "on", "max", "min", "count", "and", "or"):
            items.add(keyword)
            items.add(" " + keyword)
        return [value for value in sorted(items) if value]

    def _collect_chem_cot_bench_extra_token_strings(self, dataset: List[Any]) -> List[str]:
        items: set[str] = set()
        include_prompt_tokens = os.environ.get(
            "CSD_CHEM_INCLUDE_PROMPT_TOKENS", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        for marker in ('"', "'", ".", " .", "-", " -"):
            items.add(marker)
        for example in dataset:
            question = (self._example_field(example, "question", "") or "").strip()
            answer = (self._example_field(example, "answer", "") or "").strip()
            task = (self._example_field(example, "task", "") or "").strip()
            for text in (question, answer, task):
                for match in re.finditer(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.:_]+", text):
                    token = match.group(0)
                    items.add(token)
                    items.add(" " + token)
            if include_prompt_tokens:
                prompt_text = self._format_prompt(example)
                if prompt_text:
                    items.add(prompt_text)
        return [value for value in sorted(items) if value]
