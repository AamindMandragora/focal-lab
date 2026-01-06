# CSD Generation: Constrained Decoding Strategy Synthesis Pipeline

A synthesis pipeline for generating **Constrained Decoding Strategies (CSD)** using LLMs (Qwen) with formal verification via Dafny.

The pipeline automatically generates, verifies, compiles, and tests constrained decoding strategies that guarantee valid output from language models according to specified grammars (JSON, SQL, etc.).

## Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Generate    │────▶│   2. Verify     │────▶│   3. Compile    │────▶│    4. Test      │
│  (Qwen LLM)     │     │   (Dafny)       │     │  (Dafny → Py)   │     │   (Runtime)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │                       │
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
   Dafny Strategy         Proof Checked           Python Module          Validated Output
        │                       │                       │                       │
        └───────────────────────┴───────────────────────┴───────────────────────┘
                                        │
                                        ▼
                              Feedback Loop (on failure)
```

## Quick Start

### Prerequisites

1. **Python 3.10+** with dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dafny 4.x** (for verification and compilation):
   ```bash
   # macOS
   brew install dafny

   # Ubuntu/Debian
   # Download from: https://github.com/dafny-lang/dafny/releases

   # Verify installation
   dafny --version
   ```

3. **GPU (recommended)** for running Qwen models efficiently, or use a smaller model for CPU.

### Basic Usage

```bash
# Generate a CSD strategy for JSON output
python run_synthesis.py --task "Generate a strategy for JSON output"

# With custom max iterations
python run_synthesis.py --task "Generate a CRANE-style strategy" --max-iterations 10

# Use a smaller model for faster testing (CPU-friendly)
python run_synthesis.py --task "Generate a simple retry strategy" \
    --model Qwen/Qwen2.5-Coder-3B-Instruct

# Specify output name
python run_synthesis.py --task "Create a hybrid JSON strategy" --output-name my_strategy

# Specify location for dafny executable
python run_synthesis.py --task "Create a hybrid JSON strategy" --dafny-path ./dafny-lang/dafny/dafny
```

---

## Main Entry Point: `run_synthesis.py`

This is the CLI entry point for the synthesis pipeline.

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--task` | `-t` | (required) | Task description for strategy generation |
| `--max-iterations` | `-n` | 5 | Maximum refinement iterations |
| `--model` | `-m` | `Qwen/Qwen2.5-Coder-7B-Instruct` | HuggingFace model name |
| `--output-name` | `-o` | `generated_csd` | Name for the output module |
| `--output-dir` | | `outputs/generated-csd/` | Base output directory |
| `--dafny-path` | | `dafny` | Path to Dafny executable |
| `--temperature` | | 0.7 | Sampling temperature for Qwen |
| `--max-tokens` | | 256 | Maximum tokens to generate per attempt |
| `--device` | | `auto` | Device for inference: `cuda`, `mps`, `cpu`, `auto` |
| `--verify-only` | | | Only verify existing `GeneratedCSD.dfy` |
| `--compile-only` | | | Verify and compile without generating |
| `--no-save-reports` | | | Don't save failure/success reports |

### Examples

```bash
# Full synthesis with JSON task
python run_synthesis.py --task "Generate a constrained decoding strategy that ensures valid JSON output using hybrid generation with occasional unconstrained steps"

# Verify an existing Dafny file
python run_synthesis.py --verify-only

# Verify and compile an existing file
python run_synthesis.py --compile-only --output-name my_strategy
```

---

## Project Structure

```
csd-generation/
├── run_synthesis.py          # Main CLI entry point
├── requirements.txt          # Python dependencies
│
├── synthesis/                # Core synthesis pipeline
│   ├── generator.py          # Qwen-based strategy generation
│   ├── verifier.py           # Dafny verification wrapper
│   ├── compiler.py           # Dafny → Python compilation
│   ├── runner.py             # Runtime testing
│   ├── feedback_loop.py      # Main orchestration with feedback
│   ├── prompts.py            # LLM prompt templates
│   └── rationale.py          # Strategy rationale extraction
│
├── dafny/                    # Dafny source files
│   ├── GeneratedCSD.dfy      # Template for generated strategies
│   └── VerifiedAgentSynthesis.dfy  # Core Dafny verification module
│
├── runtime/                  # Python runtime for compiled strategies
│   └── runtime_stubs.py      # Extern implementations for Dafny code
│
├── parsers/                  # Grammar and parsing utilities
│   ├── lark_parser.py        # Generic Lark-based grammar parser
│   └── model_token_parser.py # Token-level parsing for LMs
│
├── grammars/                 # Lark grammar files
│   ├── json.lark
│   ├── sql.lark
│   ├── math.lark
│   └── gsm.lark
│
├── scripts/                  # Utility scripts
│   ├── run_csd_with_grammar.py    # Run CSD with custom grammar
│   └── validate_json_csd.py       # Validate JSON-oriented strategies
│
├── tests/                    # Unit tests
│
├── outputs/                  # Generated outputs
│   └── generated-csd/
│       ├── latest_run.txt    # Pointer to most recent run
│       └── runs/             # Individual run directories
│           └── YYYYMMDD_HHMMSS_HASH/
│               ├── generated_csd.dfy
│               ├── success_report.json (or failure_report.json)
│               └── generated_csd/      # Compiled Python module
│
└── docs/                     # Research paper and documentation
    └── paper/                # Academic paper LaTeX source
```

---

## Key Components

### 1. Strategy Generator (`synthesis/generator.py`)

The `StrategyGenerator` class uses Qwen to generate Dafny CSD strategy code.

```python
from synthesis.generator import StrategyGenerator

# Initialize with custom model
generator = StrategyGenerator(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    device="cuda",           # or "mps", "cpu", "auto"
    max_new_tokens=512,
    temperature=0.7
)

# Generate initial strategy
strategy = generator.generate_initial("Generate a JSON validation strategy")

# Refine after errors
refined = generator.refine_after_verification_error(strategy, error_message)
refined = generator.refine_after_runtime_error(strategy, traceback)
refined = generator.refine_after_compilation_error(strategy, error_message)

# Inject into Dafny template
full_dafny_code = generator.inject_strategy(strategy)
```

### 2. Dafny Verifier (`synthesis/verifier.py`)

The `DafnyVerifier` runs formal verification on generated Dafny code.

```python
from synthesis.verifier import DafnyVerifier

verifier = DafnyVerifier(
    dafny_path="dafny",
    timeout=60
)

# Verify code string
result = verifier.verify(dafny_code)
if result.success:
    print("Verification passed!")
else:
    print(result.get_error_summary())

# Verify file directly
result = verifier.verify_file(Path("my_strategy.dfy"))
```

### 3. Dafny Compiler (`synthesis/compiler.py`)

The `DafnyCompiler` compiles verified Dafny code to Python.

```python
from synthesis.compiler import DafnyCompiler

compiler = DafnyCompiler(
    dafny_path="dafny",
    output_dir=Path("outputs/"),
    timeout=120
)

result = compiler.compile(dafny_code, output_name="my_csd")
if result.success:
    print(f"Compiled to: {result.output_dir}")
    print(f"Main module: {result.main_module_path}")
```

### 4. Synthesis Pipeline (`synthesis/feedback_loop.py`)

The `SynthesisPipeline` orchestrates the full generate → verify → compile → test loop.

```python
from synthesis.feedback_loop import SynthesisPipeline, SynthesisExhaustionError

pipeline = SynthesisPipeline(
    max_iterations=5,
    save_reports=True
)

try:
    result = pipeline.synthesize(
        task_description="Generate a JSON strategy",
        output_name="json_csd"
    )
    print(f"Success! Strategy: {result.strategy_code}")
    print(f"Compiled to: {result.compiled_module_path}")
except SynthesisExhaustionError as e:
    print(e.get_failure_summary())c
```

---

## Running with Custom Grammars

Use `scripts/run_csd_with_grammar.py` to test compiled strategies with specific grammars:

```bash
# With a .lark grammar file
python scripts/run_csd_with_grammar.py \
    --run-dir outputs/generated-csd/runs/20260105_204255_8b7116 \
    --grammar grammars/json.lark

# With a built-in format
python scripts/run_csd_with_grammar.py \
    --run-dir outputs/generated-csd/runs/20260105_204255_8b7116 \
    --format json

# With HuggingFace tokenizer vocabulary
python scripts/run_csd_with_grammar.py \
    --run-dir outputs/generated-csd/runs/20260105_204255_8b7116 \
    --format json \
    --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct

# Custom options
python scripts/run_csd_with_grammar.py \
    --run-dir outputs/generated-csd/runs/XXXXX \
    --grammar grammars/sql.lark \
    --max-steps 100 \
    --vocab-size 1000 \
    --seed 42
```

### Built-in Formats

- `json` - JSON according to ECMA-404
- `sql` - Basic SELECT statements
- `math` - Mathematical expressions

---

## JSON Validation Script

Use `scripts/validate_json_csd.py` to validate JSON-oriented strategies:

```bash
# Run full validation
python scripts/validate_json_csd.py --run-dir outputs/generated-csd/runs/XXXXX

# Test only the prefix validator
python scripts/validate_json_csd.py --test-only

# Output as JSON
python scripts/validate_json_csd.py --json
```

---

## Parsers Module

The `parsers/` module provides grammar-based validation:

### Lark Grammar Parser (Recommended)

```python
from parsers import LarkGrammarParser, create_parser_for_format

# Use built-in format
parser = create_parser_for_format("json")

# Use custom grammar file
parser = LarkGrammarParser.from_grammar_file("my_grammar.lark")

# Check validity
is_valid = parser.is_valid_prefix('{"key": ')  # True
is_complete = parser.is_complete('{"key": "value"}')  # True
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_json_prefix.py -v

# Run with timeout
pytest tests/ -v --timeout=30
```

---

## Output Structure

Each synthesis run creates a unique directory:

```
outputs/generated-csd/runs/20260105_204255_8b7116/
├── generated_csd.dfy           # The Dafny source
├── success_report.json         # Metadata and rationale
└── generated_csd/              # Compiled Python module
    ├── __main__.py
    ├── _dafny/                 # Dafny runtime
    ├── GeneratedCSD.py         # Main generated module
    ├── VerifiedDecoderAgent.py # Verification support
    └── ...
```

### Success Report Format

```json
{
  "strategy_code": "// CSD_RATIONALE_BEGIN\n// Hybrid approach...\n// CSD_RATIONALE_END\ngenerated := CSDHelpers.HybridGeneration(...);",
  "tool_choice_rationale": "Hybrid approach balances creativity and validity...",
  "dafny_file": "/path/to/generated_csd.dfy",
  "compiled_dir": "/path/to/generated_csd/",
  "total_attempts": 1,
  "timestamp": "2026-01-05T20:43:18.641197"
}
```

### Failure Report Format

```json
{
  "task_description": "Generate a strategy...",
  "total_attempts": 5,
  "timestamp": "...",
  "attempts": [
    {
      "attempt_number": 1,
      "strategy_code": "...",
      "failed_at": "verification",
      "error_summary": "..."
    }
  ],
  "failure_patterns": {
    "verification_failures": 3,
    "compilation_failures": 1,
    "runtime_failures": 1
  }
}
```

---

## Available CSD Strategies

The Dafny template supports these helper functions for building strategies:

| Function | Description |
|----------|-------------|
| `CSDHelpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps)` | Fully constrained generation - always valid |
| `CSDHelpers.UnconstrainedGeneration(lm, prompt, maxSteps)` | No constraints - may produce invalid output |
| `CSDHelpers.HybridGeneration(lm, parser, prompt, maxSteps, interval)` | Alternates constrained/unconstrained every `interval` steps |
| `CSDHelpers.TryUnconstrainedThenConstrained(lm, parser, prompt, maxSteps, n)` | Try `n` unconstrained steps, fall back to constrained |
| `CSDHelpers.ConstrainedStep(lm, parser, prompt, generated)` | Single constrained step |
| `CSDHelpers.UnconstrainedStep(lm, prompt, generated)` | Single unconstrained step |

---

## Dafny Template

The generated strategies are injected into `dafny/GeneratedCSD.dfy`:

```dafny
method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat) 
  returns (generated: Prefix)
  modifies lm.Logits
  requires lm.ValidTokensIdsLogits()
  requires parser.IsValidPrefix([])
  ensures lm.ValidTokensIdsLogits()
  ensures |generated| <= maxSteps
  ensures parser.IsValidPrefix(generated)
  ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
{
  // QWEN_INSERT_STRATEGY_HERE
}
```

The postconditions guarantee:
1. Token/logit consistency is maintained
2. Output length doesn't exceed `maxSteps`
3. Output is always a valid prefix according to the grammar
4. Output is complete if it stopped before `maxSteps`

---

## Troubleshooting

### Dafny not found
```bash
# Check if dafny is in PATH
which dafny

# Or specify path explicitly
python run_synthesis.py --task "..." --dafny-path /path/to/dafny
```

### GPU memory issues
```bash
# Use a smaller model
python run_synthesis.py --task "..." --model Qwen/Qwen2.5-Coder-3B-Instruct

# Or force CPU
python run_synthesis.py --task "..." --device cpu
```

### Verification timeout
```bash
# Increase iterations (gives more chances to find a valid strategy)
python run_synthesis.py --task "..." --max-iterations 10
```

### Import errors with compiled modules
```python
# Add the output directory to Python path
import sys
sys.path.insert(0, "outputs/generated-csd/runs/XXXXX/generated_csd")
import GeneratedCSD
```

---

## License

See repository for license details.

