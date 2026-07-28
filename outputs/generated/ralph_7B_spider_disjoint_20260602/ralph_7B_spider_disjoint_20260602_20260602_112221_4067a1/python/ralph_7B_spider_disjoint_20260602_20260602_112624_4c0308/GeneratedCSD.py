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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one valid SQL query. Output format must be: SQL: <<SELECT ... FROM ...>>. Use only the schema provided. No markdown, no explanation. The query goes inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_phaseBudget_: int
        d_2_phaseBudget_ = 8
        d_3_enteredConstrained_: bool
        d_3_enteredConstrained_ = insideConstrained
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_4_phaseSteps_: int
            d_4_phaseSteps_ = 0
            while ((d_4_phaseSteps_) < (d_2_phaseBudget_)) and ((d_1_steps_) < (maxSteps)):
                d_5_next_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                d_5_next_ = out0_
                d_1_steps_ = (d_1_steps_) + (1)
                d_4_phaseSteps_ = (d_4_phaseSteps_) + (1)
                if (d_5_next_) == (eosToken):
                    d_4_phaseSteps_ = d_2_phaseBudget_
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_5_next_]))
                    if (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                        insideConstrainedOut = True
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_3_enteredConstrained_ = True
                        d_4_phaseSteps_ = d_2_phaseBudget_
                    elif (d_5_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ":"))):
                        d_4_phaseSteps_ = d_2_phaseBudget_
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_6_openGenerated_: _dafny.Seq
            d_7_openInside_: bool
            d_8_openCurrent_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_6_openGenerated_ = out1_
            d_7_openInside_ = out2_
            d_8_openCurrent_ = out3_
            generated = d_6_openGenerated_
            insideConstrainedOut = d_7_openInside_
            currentConstrainedOut = d_8_openCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_9_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_9_next_ = out4_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_9_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out5_: _dafny.Seq
                        out6_: bool
                        out7_: _dafny.Seq
                        out5_, out6_, out7_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out5_
                        d_11_closedInside_ = out6_
                        d_12_closedCurrent_ = out7_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        d_15_wasConstrained_: bool
                        out8_: _dafny.Seq
                        out9_: bool
                        out8_, out9_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_14_next_ = out8_
                        d_15_wasConstrained_ = out9_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_appendedGenerated_: _dafny.Seq
                            d_17_appendedInside_: bool
                            d_18_appendedCurrent_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_16_appendedGenerated_ = out10_
                            d_17_appendedInside_ = out11_
                            d_18_appendedCurrent_ = out12_
                            generated = d_16_appendedGenerated_
                            insideConstrainedOut = d_17_appendedInside_
                            currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

