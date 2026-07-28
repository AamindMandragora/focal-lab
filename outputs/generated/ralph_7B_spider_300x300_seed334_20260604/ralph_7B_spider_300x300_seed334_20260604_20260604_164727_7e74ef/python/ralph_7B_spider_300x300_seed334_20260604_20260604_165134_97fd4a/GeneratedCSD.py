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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<YOUR QUERY>> where YOUR QUERY is a valid SQL SELECT statement using only tables and columns from the schema. Do not output any explanation, markdown, or extra text. The query goes inside << >>.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_forcedEntry_: bool
        d_2_forcedEntry_ = False
        with _dafny.label("0"):
            while ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    if (d_1_steps_) < (maxSteps):
                        d_3_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            d_4_eg_: _dafny.Seq
                            d_5_ei_: bool
                            d_6_ec_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                            d_4_eg_ = out1_
                            d_5_ei_ = out2_
                            d_6_ec_ = out3_
                            generated = d_4_eg_
                            insideConstrainedOut = d_5_ei_
                            currentConstrainedOut = d_6_ec_
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            d_7_genLen_: int
                            d_7_genLen_ = (len(generated)) - (len(generatedPrefix))
                            if ((d_7_genLen_) >= (3)) and ((d_1_steps_) < (maxSteps)):
                                d_8_og_: _dafny.Seq
                                d_9_oi_: bool
                                d_10_oc_: _dafny.Seq
                                out4_: _dafny.Seq
                                out5_: bool
                                out6_: _dafny.Seq
                                out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                d_8_og_ = out4_
                                d_9_oi_ = out5_
                                d_10_oc_ = out6_
                                generated = d_8_og_
                                insideConstrainedOut = d_9_oi_
                                currentConstrainedOut = d_10_oc_
                                d_1_steps_ = (d_1_steps_) + (1)
                    pass
            pass
        with _dafny.label("1"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("1"):
                    if not(insideConstrainedOut):
                        d_11_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_11_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                            if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                d_12_eg_: _dafny.Seq
                                d_13_ei_: bool
                                d_14_ec_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_12_eg_ = out8_
                                d_13_ei_ = out9_
                                d_14_ec_ = out10_
                                generated = d_12_eg_
                                insideConstrainedOut = d_13_ei_
                                currentConstrainedOut = d_14_ec_
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out11_: _dafny.Seq
                        out12_: bool
                        out13_: _dafny.Seq
                        out11_, out12_, out13_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out11_
                        d_16_closedInside_ = out12_
                        d_17_closedCurrent_ = out13_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("1")
                    elif True:
                        d_18_constrainedPrompt_: _dafny.Seq
                        d_18_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_19_next_: _dafny.Seq
                        d_20_wasConstrained_: bool
                        out14_: _dafny.Seq
                        out15_: bool
                        out14_, out15_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_18_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_19_next_ = out14_
                        d_20_wasConstrained_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_19_next_) == (eosToken):
                            raise _dafny.Break("1")
                        elif True:
                            d_21_appendedGenerated_: _dafny.Seq
                            d_22_appendedInside_: bool
                            d_23_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                            d_21_appendedGenerated_ = out16_
                            d_22_appendedInside_ = out17_
                            d_23_appendedCurrent_ = out18_
                            generated = d_21_appendedGenerated_
                            insideConstrainedOut = d_22_appendedInside_
                            currentConstrainedOut = d_23_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

