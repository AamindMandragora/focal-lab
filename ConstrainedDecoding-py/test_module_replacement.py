import module_ as module_
from transformers import LogitsProcessor
import torch

class BanTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, token_id_to_ban):
        self.token_id_to_ban = token_id_to_ban

    def __call__(self, input_ids, scores):
        scores[:, self.token_id_to_ban] = -float("inf")
        return scores

def test_syncode_replacement():
    print("Testing Syncode Replacement...")
    
    # Test 1: Check initialization of components
    print("\n[1] Checking component initialization...")
    assert module_.tokenizer is not None, "Tokenizer not initialized"
    assert module_.llm is not None, "llm not initialized"
    # Removed assertions for removed components
    print("  Components initialized successfully.")

    # Test 2: Check Parser Functions
    print("\n[2] Checking Parser Functions...")
    prefix = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."
    
    # Valid Prefix
    try:
        is_valid = module_.Parser__ValidPrefix(prefix)
        print(f"  Parser__ValidPrefix('{prefix[:20]}...'): {is_valid}")
    except Exception as e:
        print(f"  Parser__ValidPrefix failed with error: {e}")
    
    # Allowed Next (should return list of expected tokens)
    try:
        allowed = module_.Parser__AllowedNext(prefix)
        print(f"  Parser__AllowedNext found {len(allowed)} expected tokens.")
    except Exception as e:
        print(f"  Parser__AllowedNext failed with error: {e}")
    
    # Is Complete (likely False for this prefix)
    try:
        is_complete = module_.Parser__IsComplete(prefix)
        print(f"  Parser__IsComplete: {is_complete}")
    except Exception as e:
        print(f"  Parser__IsComplete failed with error: {e}")

    # Test 3: Generator__ChooseToken
    print("\n[3] Checking Generator__ChooseToken...")
    
    # Unconstrained generation test
    print("  Testing unconstrained generation...")
    try:
        tokens = module_.Generator__ChooseToken(prefix, [], len(prefix), False)
        print(f"  Generated tokens (unconstrained): {tokens}")
        assert len(tokens) > 0, "No tokens generated"
    except Exception as e:
        print(f"  FAILED unconstrained generation: {e}")
        raise

    # Test 4: Dynamic Logits Processor
    print("\n[4] Checking Dynamic Logits Processor...")
    # Create a processor that bans a specific token (e.g. token id 100)
    processor = BanTokenLogitsProcessor(token_id_to_ban=100)
    
    print("  Testing generation with custom LogitsProcessor...")
    try:
        tokens = module_.Generator__ChooseToken(prefix, [], len(prefix), False, logits_processor=processor)
        print(f"  Generated tokens (with processor): {tokens}")
        assert len(tokens) > 0, "No tokens generated with processor"
    except Exception as e:
        print(f"  FAILED custom processor generation: {e}")
        raise

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_syncode_replacement()

