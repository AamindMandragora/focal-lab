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
        d_1_guidance_: _dafny.Seq
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step, but keep the reasoning concise. Put arithmetic expressions and especially the final answer inside visible << >> delimiters. Inside a delimiter span, write only a compact symbolic expression or number, then close the span immediately with >>."))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_forceAfter_: int
        d_3_forceAfter_ = 72
        d_4_stopAfter_: int
        d_4_stopAfter_ = 170
        d_5_insideTokenCap_: int
        d_5_insideTokenCap_ = 28
        d_6_forcedFinal_: bool
        d_6_forcedFinal_ = False
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_7_openCount_: int
                        out0_: int
                        out0_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                        d_7_openCount_ = out0_
                        d_8_closeCount_: int
                        out1_: int
                        out1_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
                        d_8_closeCount_ = out1_
                        if (((d_7_openCount_) > (0)) and ((d_7_openCount_) == (d_8_closeCount_))) and ((d_2_steps_) >= (d_4_stopAfter_)):
                            raise _dafny.Break("0")
                        elif (((d_7_openCount_) == (d_8_closeCount_)) and ((d_7_openCount_) == (0))) and ((d_2_steps_) >= (d_3_forceAfter_)):
                            d_9_openedGenerated_: _dafny.Seq
                            d_10_openedInside_: bool
                            d_11_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_9_openedGenerated_ = out2_
                            d_10_openedInside_ = out3_
                            d_11_openedCurrent_ = out4_
                            generated = d_9_openedGenerated_
                            insideConstrainedOut = d_10_openedInside_
                            currentConstrainedOut = d_11_openedCurrent_
                            d_6_forcedFinal_ = True
                            d_2_steps_ = (d_2_steps_) + (1)
                        elif True:
                            d_12_nextFree_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_12_nextFree_ = out5_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_12_nextFree_) == (eosToken):
                                if (((d_7_openCount_) == (d_8_closeCount_)) and ((d_7_openCount_) == (0))) and ((d_2_steps_) < (maxSteps)):
                                    d_13_eosOpenedGenerated_: _dafny.Seq
                                    d_14_eosOpenedInside_: bool
                                    d_15_eosOpenedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_13_eosOpenedGenerated_ = out6_
                                    d_14_eosOpenedInside_ = out7_
                                    d_15_eosOpenedCurrent_ = out8_
                                    generated = d_13_eosOpenedGenerated_
                                    insideConstrainedOut = d_14_eosOpenedInside_
                                    currentConstrainedOut = d_15_eosOpenedCurrent_
                                    d_6_forcedFinal_ = True
                                    d_2_steps_ = (d_2_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif (d_12_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_nextFree_]))
                                d_16_observedGenerated_: _dafny.Seq
                                d_17_observedInside_: bool
                                d_18_observedCurrent_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_16_observedGenerated_ = out9_
                                d_17_observedInside_ = out10_
                                d_18_observedCurrent_ = out11_
                                generated = d_16_observedGenerated_
                                insideConstrainedOut = d_17_observedInside_
                                currentConstrainedOut = d_18_observedCurrent_
                                d_6_forcedFinal_ = False
                            elif (d_12_nextFree_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                                pass
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_12_nextFree_]))
                    elif ((d_2_steps_) + (1)) >= (maxSteps):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_19_closedGenerated_: _dafny.Seq
                        d_20_closedInside_: bool
                        d_21_closedCurrent_: _dafny.Seq
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out12_, out13_, out14_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_19_closedGenerated_ = out12_
                        d_20_closedInside_ = out13_
                        d_21_closedCurrent_ = out14_
                        generated = d_19_closedGenerated_
                        insideConstrainedOut = d_20_closedInside_
                        currentConstrainedOut = d_21_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_6_forcedFinal_) or ((d_2_steps_) >= (d_4_stopAfter_)):
                            raise _dafny.Break("0")
                    elif (len(currentConstrainedOut)) >= (d_5_insideTokenCap_):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_6_forcedFinal_) or ((d_2_steps_) >= (d_4_stopAfter_)):
                            raise _dafny.Break("0")
                    elif True:
                        d_22_constrainedPrompt_: _dafny.Seq
                        d_22_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_23_nextConstrained_: _dafny.Seq
                        out15_: _dafny.Seq
                        out15_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_22_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_23_nextConstrained_ = out15_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_23_nextConstrained_) == (eosToken):
                            if (d_2_steps_) < (maxSteps):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_6_forcedFinal_) or ((d_2_steps_) >= (d_4_stopAfter_)):
                                    raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif (d_23_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                            insideConstrainedOut = False
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            if (d_6_forcedFinal_) or ((d_2_steps_) >= (d_4_stopAfter_)):
                                raise _dafny.Break("0")
                        elif (d_23_nextConstrained_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            if (d_2_steps_) < (maxSteps):
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                                insideConstrainedOut = False
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_steps_ = (d_2_steps_) + (1)
                                if (d_6_forcedFinal_) or ((d_2_steps_) >= (d_4_stopAfter_)):
                                    raise _dafny.Break("0")
                            elif True:
                                raise _dafny.Break("0")
                        elif True:
                            d_24_valid_: bool
                            out16_: bool
                            out16_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_23_nextConstrained_)
                            d_24_valid_ = out16_
                            if d_24_valid_:
                                d_25_appendedGenerated_: _dafny.Seq
                                d_26_appendedInside_: bool
                                d_27_appendedCurrent_: _dafny.Seq
                                out17_: _dafny.Seq
                                out18_: bool
                                out19_: _dafny.Seq
                                out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextConstrained_)
                                d_25_appendedGenerated_ = out17_
                                d_26_appendedInside_ = out18_
                                d_27_appendedCurrent_ = out19_
                                generated = d_25_appendedGenerated_
                                insideConstrainedOut = d_26_appendedInside_
                                currentConstrainedOut = d_27_appendedCurrent_
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_2_steps_) < (maxSteps)):
                                    d_28_justClosedGenerated_: _dafny.Seq
                                    d_29_justClosedInside_: bool
                                    d_30_justClosedCurrent_: _dafny.Seq
                                    out20_: _dafny.Seq
                                    out21_: bool
                                    out22_: _dafny.Seq
                                    out20_, out21_, out22_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_28_justClosedGenerated_ = out20_
                                    d_29_justClosedInside_ = out21_
                                    d_30_justClosedCurrent_ = out22_
                                    generated = d_28_justClosedGenerated_
                                    insideConstrainedOut = d_29_justClosedInside_
                                    currentConstrainedOut = d_30_justClosedCurrent_
                                    d_2_steps_ = (d_2_steps_) + (1)
                                    if (d_6_forcedFinal_) or ((d_2_steps_) >= (d_4_stopAfter_)):
                                        raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_2_steps_) < (maxSteps)):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            d_2_steps_ = (d_2_steps_) + (1)
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

