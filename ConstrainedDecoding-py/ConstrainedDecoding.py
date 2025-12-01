# ConstrainedDecoding.py
import module_ as module_
import _dafny as _dafny
import System_ as System_
import heapq
import math
import time

class default__:
    def __init__(self):
        pass

    @staticmethod
    def IsCorrectDecoder(prefix: str, result: str, maxSteps: int):
        def cond(k):
            return not ((len(prefix) <= k < len(result))) or \
                   (result[k] in module_.Parser__AllowedNext(
                        _dafny.SeqWithoutIsStrInference(result[:k])
                    ))

        all_good = _dafny.quantifier(
            _dafny.IntegerRange(len(prefix), len(result)), True, cond
        )

        return (
            module_.Parser__ValidPrefix(result)
            and (module_.Parser__IsComplete(result) or len(result) == len(prefix) + maxSteps)
            and all_good
            and len(result) >= len(prefix)
            and prefix == _dafny.SeqWithoutIsStrInference(result[:len(prefix)])
        )

    @staticmethod 
    def ConstrainedDecode(prefix: str, maxSteps: int): 
        """ 
        Standard Syncode constrained decode. Uses module_.Generator__ChooseToken which performs constrained/unconstrained switching by << >> markers. 
        """ 
        pointer = len(prefix)
        is_constrained = False
        while True: 
            with _dafny.label():
                # if module_.Parser__IsComplete(prefix): 
                #     return prefix 
                if maxSteps == 0: 
                    return prefix 
                if is_constrained:
                    allowed = module_.Parser__AllowedNext(prefix[pointer:][prefix.index("<<"):]) 
                    if len(allowed) == 0: 
                        return prefix
                else:
                    allowed = []
                tokens = module_.Generator__ChooseToken(prefix, allowed, pointer, is_constrained)
                tok_str = module_.tokenizer.convert_tokens_to_string(tokens)
                prefix = prefix + tok_str
                print(tok_str)
                if is_constrained:
                    constrained = prefix[pointer:][prefix.index("<<"):]
                    if constrained[-1] == ">>":
                        pointer = len(prefix)
                maxSteps -= 1
                raise _dafny.TailCall()