import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_
import VerifiedDecoderAgent as VerifiedDecoderAgent

# Module: GeneratedCSD

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def MyCSDStrategy(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        insideConstrainedOut: bool = False
        currentConstrainedOut: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = generatedPrefix
        insideConstrainedOut = False
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem concisely. Do any reasoning in plain text, but put the final symbolic answer expression in exactly one visible << >> span. Do not open << until the final answer expression is ready. Inside << >> use only compact arithmetic with variables, numbers, parentheses, +, -, *, /, //, and int(expr) if needed; no words or units inside the span. Close the span immediately with >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_forceAfter_: int
        d_3_forceAfter_ = 56
        d_4_maxOpenSpanTokens_: int
        d_4_maxOpenSpanTokens_ = 40
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    d_5_openCount_: int
                    out0_: int
                    out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                    d_5_openCount_ = out0_
                    d_6_closeCount_: int
                    out1_: int
                    out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                    d_6_closeCount_ = out1_
                    if (d_5_openCount_) > (d_6_closeCount_):
                        d_7_sinceOpen_: int
                        out2_: int
                        out2_ = VerifiedDecoderAgent.CSDHelpers.TokensSinceLastOccurrence(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_7_sinceOpen_ = out2_
                        if (d_7_sinceOpen_) >= (d_4_maxOpenSpanTokens_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                            d_2_steps_ = (d_2_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_8_nextInsideText_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_nextInsideText_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_8_nextInsideText_) == (eosToken):
                                if (d_2_steps_) < (maxSteps):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                    d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("0")
                            elif (d_8_nextInsideText_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_nextInsideText_]))
                                raise _dafny.Break("0")
                            elif (d_8_nextInsideText_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_nextInsideText_]))
                    elif True:
                        if (d_5_openCount_) > (0):
                            raise _dafny.Break("0")
                        elif (d_2_steps_) >= (d_3_forceAfter_):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_9_nextFree_: _dafny.Seq
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_nextFree_ = out4_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_9_nextFree_) == (eosToken):
                                if (d_2_steps_) < (maxSteps):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]))
                                    d_2_steps_ = (d_2_steps_) + (1)
                                if (d_2_steps_) < (maxSteps):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0"))]))
                                    d_2_steps_ = (d_2_steps_) + (1)
                                if (d_2_steps_) < (maxSteps):
                                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                    d_2_steps_ = (d_2_steps_) + (1)
                                raise _dafny.Break("0")
                            elif (d_9_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_nextFree_]))
                            elif (d_9_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_nextFree_]))
                    pass
            pass
        if (((((d_2_steps_) == (0)) and ((maxSteps) > (0))) and ((generated) == (generatedPrefix))) and ((insideConstrainedOut) == (insideConstrained))) and ((currentConstrainedOut) == (currentConstrained)):
            cost = 1
        elif True:
            cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

