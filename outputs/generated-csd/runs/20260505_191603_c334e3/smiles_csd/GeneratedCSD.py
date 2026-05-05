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
        d_2_chemistryArmed_: bool
        d_2_chemistryArmed_ = False
        d_3_recentAfterAnswer_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
        d_3_recentAfterAnswer_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_chemistryArmed_:
                            d_4_openedGenerated_: _dafny.Seq
                            d_5_openedInside_: bool
                            d_6_openedCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_4_openedGenerated_ = out1_
                            d_5_openedInside_ = out2_
                            d_6_openedCurrent_ = out3_
                            generated = d_4_openedGenerated_
                            insideConstrainedOut = d_5_openedInside_
                            currentConstrainedOut = d_6_openedCurrent_
                            d_2_chemistryArmed_ = False
                            out4_: _dafny.Seq
                            out4_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                            d_3_recentAfterAnswer_ = out4_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_7_next_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_7_next_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_7_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                if (d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_8_enteredGenerated_: _dafny.Seq
                                    d_9_enteredInside_: bool
                                    d_10_enteredCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_8_enteredGenerated_ = out6_
                                    d_9_enteredInside_ = out7_
                                    d_10_enteredCurrent_ = out8_
                                    generated = d_8_enteredGenerated_
                                    insideConstrainedOut = d_9_enteredInside_
                                    currentConstrainedOut = d_10_enteredCurrent_
                                    d_2_chemistryArmed_ = False
                                elif True:
                                    out9_: _dafny.Seq
                                    out9_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                                    d_3_recentAfterAnswer_ = out9_
                                    if (((((d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES")))) or ((d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles"))))) or ((d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "molecule"))))) or ((d_7_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "molecular"))))) or ((len(d_3_recentAfterAnswer_)) > (0)):
                                        d_2_chemistryArmed_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_11_closedGenerated_: _dafny.Seq
                        d_12_closedInside_: bool
                        d_13_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_11_closedGenerated_ = out10_
                        d_12_closedInside_ = out11_
                        d_13_closedCurrent_ = out12_
                        generated = d_11_closedGenerated_
                        insideConstrainedOut = d_12_closedInside_
                        currentConstrainedOut = d_13_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_chemistryArmed_ = False
                        out13_: _dafny.Seq
                        out13_ = VerifiedDecoderAgent.CSDHelpers.ExtractAfterKeyword(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "answer")))
                        d_3_recentAfterAnswer_ = out13_
                    elif True:
                        d_14_stablePrefix_: _dafny.Seq
                        d_14_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (d_14_stablePrefix_)
                        d_16_next_: _dafny.Seq
                        out14_: _dafny.Seq
                        out14_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_16_next_ = out14_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_17_appendedGenerated_: _dafny.Seq
                            d_18_appendedInside_: bool
                            d_19_appendedCurrent_: _dafny.Seq
                            out15_: _dafny.Seq
                            out16_: bool
                            out17_: _dafny.Seq
                            out15_, out16_, out17_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_17_appendedGenerated_ = out15_
                            d_18_appendedInside_ = out16_
                            d_19_appendedCurrent_ = out17_
                            generated = d_17_appendedGenerated_
                            insideConstrainedOut = d_18_appendedInside_
                            currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

