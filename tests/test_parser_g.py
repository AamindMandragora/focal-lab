import pytest
from pathlib import Path
import sys
import types
from synthesis.runner import StrategyRunner

# We only need to test if the ParseG function works within the environment setup by _create_dafny_test_environment.
# Mocking the missing _dafny import because _dafny only gets injected during run() usually via compiled path.
from unittest.mock import MagicMock
sys.modules['_dafny'] = MagicMock()

dummy_vda = types.ModuleType('VerifiedDecoderAgent')
class DummyParser:
    pass
class DummyLM:
    pass
dummy_vda.Parser = DummyParser
dummy_vda.LM = DummyLM
sys.modules['VerifiedDecoderAgent'] = dummy_vda

class TestParserG:
    """Test the ParseG function in the Parser implementations."""

    def test_jsonparser_parseg_success(self):
        """Test that JsonParser correctly parses a valid JSON string."""
        runner = StrategyRunner(parser_mode="json")
        lm, parser, prompt, max_steps = runner._create_dafny_test_environment(Path("."))

        valid_json = '{"key": "value", "list": [1, 2, 3]}'
        success = parser.ParseG(valid_json)
        
        assert success is True
        
    def test_jsonparser_parseg_failure(self):
        """Test that JsonParser fails correctly on an invalid JSON string."""
        runner = StrategyRunner(parser_mode="json")
        lm, parser, prompt, max_steps = runner._create_dafny_test_environment(Path("."))

        invalid_json = '{"key": "value", "list": [1, 2, 3'
        success = parser.ParseG(invalid_json)
        
        assert success is False
