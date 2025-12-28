import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_

# Module: CRANE

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def ValidLogitsForPrefix(logits, prefix):
        def lambda0_(forall_var_0_):
            d_0_t_: _dafny.Seq = forall_var_0_
            return not ((d_0_t_) in (default__.Parser__AllowedNext(prefix))) or ((d_0_t_) in (logits))

        return _dafny.quantifier((default__.Parser__AllowedNext(prefix)).Elements, True, lambda0_)

    @staticmethod
    def CRANE__ChooseToken(prefix, constrained, strategy):
        if constrained:
            return default__.SampleWithStrategy(default__.MaskLogits(prefix, default__.GetLogits(prefix)), default__.Parser__AllowedNext(prefix), strategy)
        elif True:
            return default__.SampleWithStrategy(default__.GetLogits(prefix), default__.Parser__AllowedNext(prefix), strategy)

    @staticmethod
    def IsConstrained(currGen, S1):
        return (S1) in (currGen)

    @staticmethod
    def AdvancePointer(constrained, newTokens, S2, pointer):
        if (constrained) and (((newTokens)[(len(newTokens)) - (1)]) == (S2)):
            return len(newTokens)
        elif True:
            return pointer

    @staticmethod
    def CRANE__Decode(tokens, maxSteps, S1, S2, pointer, strategy):
        while True:
            with _dafny.label():
                if default__.Parser__IsComplete(tokens):
                    return tokens
                elif (maxSteps) == (0):
                    return tokens
                elif True:
                    d_0_currGen_ = _dafny.SeqWithoutIsStrInference((tokens)[pointer::])
                    d_1_constrained_ = default__.IsConstrained(d_0_currGen_, S1)
                    if (len(default__.Parser__AllowedNext(tokens))) == (0):
                        return tokens
                    elif True:
                        d_2_tok_ = default__.CRANE__ChooseToken(tokens, d_1_constrained_, strategy)
                        d_3_newTokens_ = (tokens) + (_dafny.SeqWithoutIsStrInference([d_2_tok_]))
                        d_4_pointer_ = default__.AdvancePointer(d_1_constrained_, d_3_newTokens_, S2, pointer)
                        if (d_2_tok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EOS"))):
                            return d_3_newTokens_
                        elif True:
                            in0_ = d_3_newTokens_
                            in1_ = (maxSteps) - (1)
                            in2_ = S1
                            in3_ = S2
                            in4_ = d_4_pointer_
                            in5_ = strategy
                            tokens = in0_
                            maxSteps = in1_
                            S1 = in2_
                            S2 = in3_
                            pointer = in4_
                            strategy = in5_
                            raise _dafny.TailCall()
                break

    @staticmethod
    def IsCorrectCRANEDecoder(prefix, result, maxSteps):
        def lambda0_(forall_var_0_):
            d_0_k_: int = forall_var_0_
            return not (((len(prefix)) <= (d_0_k_)) and ((d_0_k_) < (len(result)))) or (((result)[d_0_k_]) in (default__.Parser__AllowedNext(_dafny.SeqWithoutIsStrInference((result)[:d_0_k_:]))))

        return ((((default__.Parser__ValidPrefix(result)) and (((len(result)) >= (len(prefix))) and ((len(prefix)) > (0)))) and (((default__.Parser__IsComplete(result)) or ((len(result)) == ((len(prefix)) + (maxSteps)))) or (((result)[(len(result)) - (1)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "EOS")))))) and ((prefix) == (_dafny.SeqWithoutIsStrInference((result)[:len(prefix):])))) and (_dafny.quantifier(_dafny.IntegerRange(len(prefix), len(result)), True, lambda0_))


class SamplingStrategy:
    @_dafny.classproperty
    def AllSingletonConstructors(cls):
        return [SamplingStrategy_ArgMax(), SamplingStrategy_Temperature(), SamplingStrategy_TopK(), SamplingStrategy_Nucleus()]
    @classmethod
    def default(cls, ):
        return lambda: SamplingStrategy_ArgMax()
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_ArgMax(self) -> bool:
        return isinstance(self, SamplingStrategy_ArgMax)
    @property
    def is_Temperature(self) -> bool:
        return isinstance(self, SamplingStrategy_Temperature)
    @property
    def is_TopK(self) -> bool:
        return isinstance(self, SamplingStrategy_TopK)
    @property
    def is_Nucleus(self) -> bool:
        return isinstance(self, SamplingStrategy_Nucleus)

class SamplingStrategy_ArgMax(SamplingStrategy, NamedTuple('ArgMax', [])):
    def __dafnystr__(self) -> str:
        return f'CRANE.SamplingStrategy.ArgMax'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, SamplingStrategy_ArgMax)
    def __hash__(self) -> int:
        return super().__hash__()

class SamplingStrategy_Temperature(SamplingStrategy, NamedTuple('Temperature', [])):
    def __dafnystr__(self) -> str:
        return f'CRANE.SamplingStrategy.Temperature'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, SamplingStrategy_Temperature)
    def __hash__(self) -> int:
        return super().__hash__()

class SamplingStrategy_TopK(SamplingStrategy, NamedTuple('TopK', [])):
    def __dafnystr__(self) -> str:
        return f'CRANE.SamplingStrategy.TopK'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, SamplingStrategy_TopK)
    def __hash__(self) -> int:
        return super().__hash__()

class SamplingStrategy_Nucleus(SamplingStrategy, NamedTuple('Nucleus', [])):
    def __dafnystr__(self) -> str:
        return f'CRANE.SamplingStrategy.Nucleus'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, SamplingStrategy_Nucleus)
    def __hash__(self) -> int:
        return super().__hash__()

