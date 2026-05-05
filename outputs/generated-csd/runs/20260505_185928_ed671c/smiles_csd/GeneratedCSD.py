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
        d_2_chemistryCueSeen_: bool
        d_2_chemistryCueSeen_ = False
        d_3_recentAfterAnswer_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
        d_3_recentAfterAnswer_ = out0_
        d_4_flatGroups_: _dafny.Seq
        out1_: _dafny.Seq
        out1_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_4_flatGroups_ = out1_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        out2_: _dafny.Seq
                        out2_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                        d_3_recentAfterAnswer_ = out2_
                        if (not(d_2_chemistryCueSeen_)) and ((len(d_3_recentAfterAnswer_)) > (0)):
                            d_2_chemistryCueSeen_ = True
                        if d_2_chemistryCueSeen_:
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out3_
                            d_6_openedInside_ = out4_
                            d_7_openedCurrent_ = out5_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_chemistryCueSeen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out6_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_chemistryCueSeen_ = False
                                elif (((((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "molecule"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "structure")))):
                                    d_2_chemistryCueSeen_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_closedGenerated_: _dafny.Seq
                        d_10_closedInside_: bool
                        d_11_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_closedGenerated_ = out7_
                        d_10_closedInside_ = out8_
                        d_11_closedCurrent_ = out9_
                        generated = d_9_closedGenerated_
                        insideConstrainedOut = d_10_closedInside_
                        currentConstrainedOut = d_11_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_stablePrefix_: _dafny.Seq
                        d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (d_12_stablePrefix_)
                        d_14_validCount_: int
                        out10_: int
                        out10_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_14_validCount_ = out10_
                        d_15_next_: _dafny.Seq
                        d_15_next_ = eosToken
                        if (d_14_validCount_) <= (12):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_15_next_ = out11_
                        elif (len(d_4_flatGroups_)) > (0):
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, d_4_flatGroups_, _dafny.BigRational('3e0'), eosToken)
                            d_15_next_ = out12_
                        elif True:
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_15_next_ = out13_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_appendedGenerated_: _dafny.Seq
                            d_17_appendedInside_: bool
                            d_18_appendedCurrent_: _dafny.Seq
                            out14_: _dafny.Seq
                            out15_: bool
                            out16_: _dafny.Seq
                            out14_, out15_, out16_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_16_appendedGenerated_ = out14_
                            d_17_appendedInside_ = out15_
                            d_18_appendedCurrent_ = out16_
                            generated = d_16_appendedGenerated_
                            insideConstrainedOut = d_17_appendedInside_
                            currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

