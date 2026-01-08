#!/usr/bin/env python3
"""
Run a compiled CSD strategy with a specific grammar using a reusable LM, tokenizer, and parser.
"""

import argparse
import sys
import json
import random
from pathlib import Path
from typing import Optional
from math import inf

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def create_lark_dafny_parser(grammar_source: str, VerifiedDecoderAgent, _dafny, start: str = "start"):
    """
    Create a Dafny-compatible parser from a Lark grammar.
    
    Args:
        grammar_source: Either a grammar string or path to .lark file
        VerifiedDecoderAgent: The imported Dafny module
        _dafny: The Dafny runtime module
        start: Start rule name
    """
    from lark import Lark
    from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF
    
    # Load grammar - check if it's a file path (short string without newlines)
    if '\n' not in grammar_source and len(grammar_source) < 500:
        grammar_path = Path(grammar_source)
        if grammar_path.exists():
            grammar = grammar_path.read_text()
        else:
            grammar = grammar_source
    else:
        grammar = grammar_source
    
    # Create Lark parser
    lark_parser = Lark(grammar, start=start, parser='lalr')
    
    class LarkDafnyParser(VerifiedDecoderAgent.Parser):
        """Parser using Lark grammar, compatible with Dafny-compiled code."""
        
        def __init__(self, lm_tokens):
            super().__init__()
            self._lm_tokens = lm_tokens
            self._token_list = list(lm_tokens)
            self._lark = lark_parser
            self._UnexpectedCharacters = UnexpectedCharacters
            self._UnexpectedToken = UnexpectedToken
            self._UnexpectedEOF = UnexpectedEOF
        
        def _tokens_to_text(self, tokens) -> str:
            """Convert Dafny token sequence to text."""
            try:
                return "".join(str(tokens[i]) for i in range(len(tokens)))
            except (TypeError, AttributeError):
                return str(tokens)
        
        def _is_valid_prefix(self, text: str) -> bool:
            """Check if text is a valid prefix of the grammar."""
            if not text:
                return True
            try:
                self._lark.parse(text)
                return True
            except self._UnexpectedEOF:
                # Hit end of input while expecting more - valid prefix
                return True
            except self._UnexpectedToken as e:
                if e.token.type == '$END':
                    return True
                return False
            except self._UnexpectedCharacters:
                return False
            except Exception:
                return False
        
        def _is_complete(self, text: str) -> bool:
            """Check if text is a complete valid parse."""
            if not text:
                return False
            try:
                self._lark.parse(text)
                return True
            except Exception:
                return False
        
        def IsValidPrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is valid."""
            if len(prefix) == 0:
                return True
            text = self._tokens_to_text(prefix)
            return self._is_valid_prefix(text)
        
        def IsCompletePrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is complete."""
            if len(prefix) == 0:
                return False
            text = self._tokens_to_text(prefix)
            return self._is_complete(text)
        
        def ValidNextTokens(self, prefix):
            """Dafny interface: Get valid next tokens."""
            current_text = self._tokens_to_text(prefix) if len(prefix) > 0 else ""
            
            if current_text and not self._is_valid_prefix(current_text):
                return _dafny.SeqWithoutIsStrInference([])
            
            valid = []
            for token in self._token_list:
                token_str = str(token)
                if not token_str:
                    continue
                extended = current_text + token_str
                if self._is_valid_prefix(extended):
                    valid.append(token)
            
            return _dafny.SeqWithoutIsStrInference(valid)
    
    return LarkDafnyParser

def create_vocabulary(vocab_type: str = "default", tokenizer_name: Optional[str] = None, size: int = 500) -> list[str]:
    """Create vocabulary for the LM."""
    if vocab_type == "tokenizer" and tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        vocab = []
        # for i in range(min(len(tokenizer), size)):
        for i in range(len(tokenizer)):
            try:
                token = tokenizer.decode([i])
                if token:
                    vocab.append(token)
            except:
                pass
        return vocab
    
    # Default vocabulary with common tokens
    vocab = list('{}[]():,."\'+-*/=<>!&|^~%@#$_\\;? \t\n')
    vocab.extend(list('0123456789'))
    vocab.extend(list('abcdefghijklmnopqrstuvwxyz'))
    vocab.extend(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    vocab.extend(['true', 'false', 'null', 'SELECT', 'FROM', 'WHERE', 'AND', 'OR'])
    vocab.append('<EOS>')
    
    while len(vocab) < size:
        vocab.append(f'<T{len(vocab)}>')
    
    return vocab[:size]

class CSDRunner:
    """
    Encapsulates LM, tokenizer, parser, and compiled CSD module
    for repeated prompt evaluation without reloading.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.lm = None
        self.tokenizer = None
        self.parser_class = None
        self._dafny = None
        self.VerifiedDecoderAgent = None
        self.GeneratedCSD = None
        self.vocab = None

    def initialize(
        self,
        run_dir: Path,
        grammar_source: str,
        lm_name: str,
        device: str = "cpu",
        start_rule: str = "start"
    ):
        """
        Load the LM, tokenizer, vocabulary, parser, and compiled CSD module once.
        """

        if lm_name is None: 
            if tokenizer_name is None: 
                return { "success": False, "error": "Either lm_name or tokenizer_name must be provided" } 
            lm_name = tokenizer_name 

        run_dir = Path(run_dir) 
        module_dir = run_dir / "generated_csd" 

        if not module_dir.exists(): 
            return {"success": False, "error": f"Module directory not found: {module_dir}"} 
        
        if str(module_dir) not in sys.path: 
            sys.path.insert(0, str(module_dir))

        import _dafny
        import VerifiedDecoderAgent
        import GeneratedCSD

        self._dafny = _dafny
        self.VerifiedDecoderAgent = VerifiedDecoderAgent
        self.GeneratedCSD = GeneratedCSD

        # Load grammar
        if Path(grammar_source).exists():
            grammar = Path(grammar_source).read_text()

        # Build vocabulary
        self.vocab = create_vocabulary(
            vocab_type="tokenizer" if lm_name else "default",
            tokenizer_name=lm_name,
        )

        # Create grammar-backed parser
        self.parser_class = create_lark_dafny_parser(
            grammar, VerifiedDecoderAgent, _dafny, start_rule
        )

        # Load LM and tokenizer
        class HFBackedLM(VerifiedDecoderAgent.LM):
            MASKED_VAL = -1e4

            def __init__(self_inner, lm_name: str, tokens, device: str = "cpu"):
                super().__init__()

                self_inner.tokenizer = AutoTokenizer.from_pretrained(
                    lm_name, trust_remote_code=True
                )
                self_inner.model = AutoModelForCausalLM.from_pretrained(
                    lm_name,
                    trust_remote_code=True,
                    device_map=device,
                    torch_dtype=torch.float16,
                )
                self_inner.model.eval()

                self_inner._Tokens = _dafny.SeqWithoutIsStrInference(tokens)
                self_inner.Logits = _dafny.Array(None, len(tokens))

                vocab = self_inner.tokenizer.get_vocab()

                hf_ids = []
                valid = []
                for t in tokens:
                    if t in vocab:
                        hf_ids.append(vocab[t])
                        valid.append(True)
                    else:
                        hf_ids.append(0)
                        valid.append(False)

                self_inner.hf_ids = torch.tensor(hf_ids, device=self_inner.model.device)
                self_inner.valid_mask = torch.tensor(valid, device=self_inner.model.device)

                self_inner.reset()

                masked = _dafny.BigRational(str(self_inner.MASKED_VAL))
                for i in range(len(tokens)):
                    self_inner.Logits[i] = masked

            # 🔴 MUST be called per prompt
            def reset(self_inner):
                self_inner.past_kv = None
                self_inner.last_input_id = None

            def GenerateLogits(self_inner, input_prefix):
                # First call: seed from full prefix
                if self_inner.past_kv is None:
                    text = "".join(str(input_prefix[i]) for i in range(len(input_prefix)))
                    input_ids = self_inner.tokenizer(
                        text, return_tensors="pt"
                    ).input_ids.to(self_inner.model.device)
                else:
                    input_ids = torch.tensor(
                        [[self_inner.last_input_id]],
                        device=self_inner.model.device,
                    )

                with torch.no_grad():
                    out = self_inner.model(
                        input_ids=input_ids,
                        use_cache=True,
                        past_key_values=self_inner.past_kv,
                    )

                self_inner.past_kv = out.past_key_values
                logits = out.logits[0, -1]

                projected = logits[self_inner.hf_ids]
                projected = projected.masked_fill(
                    ~self_inner.valid_mask,
                    torch.finfo(projected.dtype).min,
                )

                vals = projected.float().cpu().tolist()
                for i, v in enumerate(vals):
                    self_inner.Logits[i] = _dafny.BigRational(v)

            def ChooseNextToken(self_inner):
                best_idx = 0
                best_val = self_inner.MASKED_VAL

                for i in range(self_inner.Logits.length(0)):
                    v = float(self_inner.Logits[i])
                    if v > best_val:
                        best_val = v
                        best_idx = i

                tok = self_inner._Tokens[best_idx]
                hf_id = self_inner.tokenizer.get_vocab().get(tok)

                self_inner.last_input_id = hf_id
                return tok

        self.lm = HFBackedLM(lm_name, self.vocab, device)
        self.parser = self.parser_class(self.vocab)

    def run_prompt(self, prompt: str, max_steps: int = 50):
        """
        Run a prompt using the preloaded LM, tokenizer, and parser.
        """
        if self.lm is None or self.parser is None:
            raise RuntimeError("CSDRunner not initialized. Call .initialize() first.")

        prompt_seq = self._dafny.SeqWithoutIsStrInference(prompt.split(" "))

        print(f"starting with prompt: {prompt}")

        output = self.GeneratedCSD.default__.MyCSDStrategy(
            self.lm, self.parser, prompt_seq, max_steps
        )

        print("ended prompt")

        self.lm.reset()

        output_tokens = [str(t) for t in output]
        output_text = "".join(output_tokens)

        return {
            "success": True,
            "output_tokens": output_tokens,
            "output_text": output_text,
            "output_length": len(output_tokens),
            "IsValidPrefix": self.parser.IsValidPrefix(output_text),
            "IsCompletePrefix": self.parser.IsCompletePrefix(output_text),
        }


# ------------------------------
# CLI entry
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run compiled CSD with grammar")
    parser.add_argument("--run-dir", "-r", type=Path, required=True)
    grammar_group = parser.add_mutually_exclusive_group(required=True)
    grammar_group.add_argument("--grammar", "-g", type=str)
    grammar_group.add_argument("--format", "-f", type=str, choices=["json", "sql", "math"])
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--tokenizer", "-t", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-rule", type=str, default="start")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    grammar_source = args.grammar if args.grammar else args.format
    lm_name = args.tokenizer or args.format

    random.seed(args.seed)

    runner = CSDRunner.get_instance()
    runner.initialize(
        run_dir=args.run_dir,
        grammar_source=grammar_source,
        lm_name=lm_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        start_rule=args.start_rule
    )

    results = runner.run_prompt(prompt="", max_steps=args.max_steps)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        if results["success"]:
            print(f"✓ Generation successful")
            print(f"  Output length: {results['output_length']}")
            print(f"  Output: {repr(results['output_text'][:80])}...")
        else:
            print(f"✗ Generation failed: {results.get('error', 'Unknown error')}")

    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()