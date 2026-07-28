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
        d_1_guidance_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly one SQL query in the format: SQL: <<YOUR QUERY>>. Use only the tables and columns from the provided schema. No explanation, no markdown, no extra text. Write the SQL query directly after SQL: <<"))
        (d_0_helpers_).AppendTaskGuidance(lm, d_1_guidance_)
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_prefixPhaseSteps_: int
        d_3_prefixPhaseSteps_ = 0
        d_4_maxPrefixSteps_: int
        d_4_maxPrefixSteps_ = 8
        d_5_spanForced_: bool
        d_5_spanForced_ = insideConstrained
        with _dafny.label("0"):
            while (d_2_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_3_prefixPhaseSteps_) < (d_4_maxPrefixSteps_):
                            d_6_next_: _dafny.Seq
                            out0_: _dafny.Seq
                            out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_6_next_ = out0_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_prefixPhaseSteps_ = (d_3_prefixPhaseSteps_) + (1)
                            if (d_6_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                                if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_5_spanForced_ = True
                        elif True:
                            d_7_openGenerated_: _dafny.Seq
                            d_8_openInside_: bool
                            d_9_openCurrent_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_7_openGenerated_ = out1_
                            d_8_openInside_ = out2_
                            d_9_openCurrent_ = out3_
                            generated = d_7_openGenerated_
                            insideConstrainedOut = d_8_openInside_
                            currentConstrainedOut = d_9_openCurrent_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_5_spanForced_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_10_closedGenerated_: _dafny.Seq
                        d_11_closedInside_: bool
                        d_12_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_10_closedGenerated_ = out4_
                        d_11_closedInside_ = out5_
                        d_12_closedCurrent_ = out6_
                        generated = d_10_closedGenerated_
                        insideConstrainedOut = d_11_closedInside_
                        currentConstrainedOut = d_12_closedCurrent_
                        d_2_steps_ = (d_2_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_13_constrainedPrompt_: _dafny.Seq
                        d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_14_next_: _dafny.Seq
                        d_15_wasConstrained_: bool
                        out7_: _dafny.Seq
                        out8_: bool
                        out7_, out8_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_14_next_ = out7_
                        d_15_wasConstrained_ = out8_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_14_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_appendedGenerated_: _dafny.Seq
                            d_17_appendedInside_: bool
                            d_18_appendedCurrent_: _dafny.Seq
                            out9_: _dafny.Seq
                            out10_: bool
                            out11_: _dafny.Seq
                            out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                            d_16_appendedGenerated_ = out9_
                            d_17_appendedInside_ = out10_
                            d_18_appendedCurrent_ = out11_
                            generated = d_16_appendedGenerated_
                            insideConstrainedOut = d_17_appendedInside_
                            currentConstrainedOut = d_18_appendedCurrent_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

