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
            if ((((d_3_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_3_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES"))))) or ((d_3_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles"))))) or ((d_3_lastTok_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))):
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
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_chemistryCueSeen_ = False
                                elif ((((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":")))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SMILES"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "smiles"))))) or ((d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))):
                                    d_2_chemistryCueSeen_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_closedGenerated_: _dafny.Seq
                        d_10_closedInside_: bool
                        d_11_closedCurrent_: _dafny.Seq
                        out6_: _dafny.Seq
                        out7_: bool
                        out8_: _dafny.Seq
                        out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_closedGenerated_ = out6_
                        d_10_closedInside_ = out7_
                        d_11_closedCurrent_ = out8_
                        generated = d_9_closedGenerated_
                        insideConstrainedOut = d_10_closedInside_
                        currentConstrainedOut = d_11_closedCurrent_
                        d_2_chemistryCueSeen_ = False
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_next_: _dafny.Seq
                        out9_: _dafny.Seq
                        out9_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_13_next_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_14_appendedGenerated_: _dafny.Seq
                            d_15_appendedInside_: bool
                            d_16_appendedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_14_appendedGenerated_ = out10_
                            d_15_appendedInside_ = out11_
                            d_16_appendedCurrent_ = out12_
                            generated = d_14_appendedGenerated_
                            insideConstrainedOut = d_15_appendedInside_
                            currentConstrainedOut = d_16_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

