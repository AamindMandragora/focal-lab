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
        d_3_lastTok_: _dafny.Seq
        d_4_foundLast_: bool
        out0_: _dafny.Seq
        out1_: bool
        out0_, out1_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")))
        d_3_lastTok_ = out0_
        d_4_foundLast_ = out1_
        if d_4_foundLast_:
            if ((((((d_3_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES")))) or ((d_3_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles"))))) or ((d_3_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "molecule"))))) or ((d_3_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "molecular"))))) or ((d_3_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "chemistry"))))) or ((d_3_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))):
                d_2_chemistryCueSeen_ = True
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if d_2_chemistryCueSeen_:
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out2_
                            d_6_openedInside_ = out3_
                            d_7_openedCurrent_ = out4_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_chemistryCueSeen_ = False
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_9_observedGenerated_: _dafny.Seq
                                    d_10_observedInside_: bool
                                    d_11_observedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_9_observedGenerated_ = out6_
                                    d_10_observedInside_ = out7_
                                    d_11_observedCurrent_ = out8_
                                    generated = d_9_observedGenerated_
                                    insideConstrainedOut = d_10_observedInside_
                                    currentConstrainedOut = d_11_observedCurrent_
                                    d_2_chemistryCueSeen_ = False
                                elif (((((((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES")))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "molecule"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "molecular"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "chemistry"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))):
                                    d_2_chemistryCueSeen_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_12_closedGenerated_: _dafny.Seq
                        d_13_closedInside_: bool
                        d_14_closedCurrent_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_12_closedGenerated_ = out9_
                        d_13_closedInside_ = out10_
                        d_14_closedCurrent_ = out11_
                        generated = d_12_closedGenerated_
                        insideConstrainedOut = d_13_closedInside_
                        currentConstrainedOut = d_14_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_15_constrainedPrompt_: _dafny.Seq
                        d_15_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_16_next_: _dafny.Seq
                        out12_: _dafny.Seq
                        out12_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_15_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_16_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_16_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_17_appendedGenerated_: _dafny.Seq
                            d_18_appendedInside_: bool
                            d_19_appendedCurrent_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_16_next_)
                            d_17_appendedGenerated_ = out13_
                            d_18_appendedInside_ = out14_
                            d_19_appendedCurrent_ = out15_
                            generated = d_17_appendedGenerated_
                            insideConstrainedOut = d_18_appendedInside_
                            currentConstrainedOut = d_19_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

