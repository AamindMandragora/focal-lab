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
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve step by step. Write the final numeric answer inside << >> delimiters. Example: <<n1 * p1>>. Use only one final << >> span for the answer.")))
        while (d_1_steps_) < (maxSteps):
            if not(insideConstrainedOut):
                d_2_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_2_next_ = out0_
                d_1_steps_ = (d_1_steps_) + (1)
                if (d_2_next_) == (eosToken):
                    cost = d_1_steps_
                    return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif (d_2_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                    d_3_g2_: _dafny.Seq
                    d_4_ins2_: bool
                    d_5_cur2_: _dafny.Seq
                    out1_: _dafny.Seq
                    out2_: bool
                    out3_: _dafny.Seq
                    out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                    d_3_g2_ = out1_
                    d_4_ins2_ = out2_
                    d_5_cur2_ = out3_
                    generated = d_3_g2_
                    insideConstrainedOut = d_4_ins2_
                    currentConstrainedOut = d_5_cur2_
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
            elif True:
                d_6_isComplete_: bool
                d_6_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                if d_6_isComplete_:
                    if ((d_1_steps_) + (1)) <= (maxSteps):
                        d_7_g3_: _dafny.Seq
                        d_8_ins3_: bool
                        d_9_cur3_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_g3_ = out4_
                        d_8_ins3_ = out5_
                        d_9_cur3_ = out6_
                        generated = d_7_g3_
                        insideConstrainedOut = d_8_ins3_
                        currentConstrainedOut = d_9_cur3_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        cost = d_1_steps_
                        return generated, insideConstrainedOut, currentConstrainedOut, cost
                elif True:
                    d_10_next_: _dafny.Seq
                    d_11_wasConstrained_: bool
                    out7_: _dafny.Seq
                    out8_: bool
                    out7_, out8_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                    d_10_next_ = out7_
                    d_11_wasConstrained_ = out8_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_10_next_) == (eosToken):
                        cost = d_1_steps_
                        return generated, insideConstrainedOut, currentConstrainedOut, cost
                    elif True:
                        d_12_valid_: bool
                        out9_: bool
                        out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_10_next_)
                        d_12_valid_ = out9_
                        d_13_alreadyComplete_: bool
                        d_13_alreadyComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if (d_12_valid_) and (not(d_13_alreadyComplete_)):
                            d_14_g6_: _dafny.Seq
                            d_15_ins6_: bool
                            d_16_cur6_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_14_g6_ = out10_
                            d_15_ins6_ = out11_
                            d_16_cur6_ = out12_
                            generated = d_14_g6_
                            insideConstrainedOut = d_15_ins6_
                            currentConstrainedOut = d_16_cur6_
                            d_17_nowComplete_: bool
                            d_17_nowComplete_ = (parser).IsCompletePrefix(d_16_cur6_)
                            if (d_17_nowComplete_) and (((d_1_steps_) + (1)) <= (maxSteps)):
                                d_18_g7_: _dafny.Seq
                                d_19_ins7_: bool
                                d_20_cur7_: _dafny.Seq
                                out13_: _dafny.Seq
                                out14_: bool
                                out15_: _dafny.Seq
                                out13_, out14_, out15_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_18_g7_ = out13_
                                d_19_ins7_ = out14_
                                d_20_cur7_ = out15_
                                generated = d_18_g7_
                                insideConstrainedOut = d_19_ins7_
                                currentConstrainedOut = d_20_cur7_
                                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

