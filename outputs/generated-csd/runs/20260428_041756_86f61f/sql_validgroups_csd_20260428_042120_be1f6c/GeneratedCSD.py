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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        if not(insideConstrainedOut):
            insideConstrainedOut = True
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_narrowThreshold_: int
        d_2_narrowThreshold_ = 12
        d_3_schemaBias_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_schemaBias_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        raise _dafny.Break("0")
                    elif True:
                        d_4_stablePrefix_: _dafny.Seq
                        d_4_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_5_constrainedPrompt_: _dafny.Seq
                        d_5_constrainedPrompt_ = (prompt) + (d_4_stablePrefix_)
                        d_6_validCount_: int
                        out1_: int
                        out1_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_6_validCount_ = out1_
                        if (d_6_validCount_) <= (d_2_narrowThreshold_):
                            d_7_next_: _dafny.Seq
                            out2_: _dafny.Seq
                            out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_5_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_7_next_ = out2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_8_appendedGenerated_: _dafny.Seq
                                d_9_appendedInside_: bool
                                d_10_appendedCurrent_: _dafny.Seq
                                out3_: _dafny.Seq
                                out4_: bool
                                out5_: _dafny.Seq
                                out3_, out4_, out5_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_7_next_)
                                d_8_appendedGenerated_ = out3_
                                d_9_appendedInside_ = out4_
                                d_10_appendedCurrent_ = out5_
                                generated = d_8_appendedGenerated_
                                insideConstrainedOut = d_9_appendedInside_
                                currentConstrainedOut = d_10_appendedCurrent_
                        elif True:
                            (lm).GenerateLogits((d_5_constrainedPrompt_) + (currentConstrainedOut))
                            d_11_candidates_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_5_constrainedPrompt_, currentConstrainedOut, 48, eosToken)
                            d_11_candidates_ = out6_
                            d_12_schemaPreferred_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = VerifiedDecoderAgent.CSDHelpers.IntersectTokenSets(d_11_candidates_, d_3_schemaBias_)
                            d_12_schemaPreferred_ = out7_
                            if (len(d_12_schemaPreferred_)) > (0):
                                (d_0_helpers_).BoostTokenLogits(lm, d_12_schemaPreferred_, _dafny.BigRational('8e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_13_next2_: _dafny.Seq
                            out8_: _dafny.Seq
                            out8_ = (lm).ChooseNextToken()
                            d_13_next2_ = out8_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_13_next2_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_14_appendedGenerated2_: _dafny.Seq
                                d_15_appendedInside2_: bool
                                d_16_appendedCurrent2_: _dafny.Seq
                                out9_: _dafny.Seq
                                out10_: bool
                                out11_: _dafny.Seq
                                out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next2_)
                                d_14_appendedGenerated2_ = out9_
                                d_15_appendedInside2_ = out10_
                                d_16_appendedCurrent2_ = out11_
                                generated = d_14_appendedGenerated2_
                                insideConstrainedOut = d_15_appendedInside2_
                                currentConstrainedOut = d_16_appendedCurrent2_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

