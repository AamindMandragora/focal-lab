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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output SQL: <<query>> where query is a valid SQL SELECT statement using ONLY the exact table and column names from the provided schema. The SQL must be placed between << and >>. No explanation, no markdown, no aliases not in the schema.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (((d_1_steps_) < (maxSteps)) and ((d_1_steps_) < (5))) and (not(insideConstrainedOut)):
                with _dafny.c_label("0"):
                    d_2_next_: _dafny.Seq
                    out0_: _dafny.Seq
                    out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                    d_2_next_ = out0_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_2_next_) == (eosToken):
                        raise _dafny.Break("0")
                    elif True:
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_2_next_]))
                        if VerifiedDecoderAgent.default__.RenderedEndsWith(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    pass
            pass
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_3_og_: _dafny.Seq
            d_4_oi_: bool
            d_5_oc_: _dafny.Seq
            out1_: _dafny.Seq
            out2_: bool
            out3_: _dafny.Seq
            out1_, out2_, out3_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_3_og_ = out1_
            d_4_oi_ = out2_
            d_5_oc_ = out3_
            generated = d_3_og_
            insideConstrainedOut = d_4_oi_
            currentConstrainedOut = d_5_oc_
            d_1_steps_ = (d_1_steps_) + (1)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_6_remaining_: int
            d_6_remaining_ = (maxSteps) - (d_1_steps_)
            d_7_sqlBudget_: int = int(0)
            if (d_6_remaining_) <= (1):
                d_7_sqlBudget_ = 0
            elif ((d_6_remaining_) - (1)) <= (300):
                d_7_sqlBudget_ = (d_6_remaining_) - (1)
            elif True:
                d_7_sqlBudget_ = 300
            d_8_constrainedPrompt_: _dafny.Seq
            d_8_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
            d_9_sqlSteps_: int
            d_9_sqlSteps_ = 0
            with _dafny.label("2_0"):
                while ((d_9_sqlSteps_) < (d_7_sqlBudget_)) and (insideConstrainedOut):
                    with _dafny.c_label("2_0"):
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            raise _dafny.Break("2_0")
                        d_10_next_: _dafny.Seq
                        out4_: _dafny.Seq
                        out4_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_8_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_10_next_ = out4_
                        d_9_sqlSteps_ = (d_9_sqlSteps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("2_0")
                        elif True:
                            d_11_valid_: bool
                            out5_: bool
                            out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_10_next_)
                            d_11_valid_ = out5_
                            if d_11_valid_:
                                d_12_ag_: _dafny.Seq
                                d_13_ai_: bool
                                d_14_ac_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: bool
                                out8_: _dafny.Seq
                                out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                d_12_ag_ = out6_
                                d_13_ai_ = out7_
                                d_14_ac_ = out8_
                                generated = d_12_ag_
                                insideConstrainedOut = d_13_ai_
                                currentConstrainedOut = d_14_ac_
                        pass
                pass
            d_1_steps_ = (d_1_steps_) + (d_9_sqlSteps_)
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            if (parser).IsCompletePrefix(currentConstrainedOut):
                d_15_cg_: _dafny.Seq
                d_16_ci_: bool
                d_17_cc_: _dafny.Seq
                out9_: _dafny.Seq
                out10_: bool
                out11_: _dafny.Seq
                out9_, out10_, out11_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_15_cg_ = out9_
                d_16_ci_ = out10_
                d_17_cc_ = out11_
                generated = d_15_cg_
                insideConstrainedOut = d_16_ci_
                currentConstrainedOut = d_17_cc_
                d_1_steps_ = (d_1_steps_) + (1)
            elif True:
                d_18_remaining_: int
                d_18_remaining_ = (maxSteps) - (d_1_steps_)
                d_19_closeBudget_: int = int(0)
                if (d_18_remaining_) <= (120):
                    d_19_closeBudget_ = d_18_remaining_
                elif True:
                    d_19_closeBudget_ = 120
                d_20_cg_: _dafny.Seq
                d_21_ci_: bool
                d_22_cc_: _dafny.Seq
                out12_: _dafny.Seq
                out13_: bool
                out14_: _dafny.Seq
                out12_, out13_, out14_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_19_closeBudget_)
                d_20_cg_ = out12_
                d_21_ci_ = out13_
                d_22_cc_ = out14_
                generated = d_20_cg_
                insideConstrainedOut = d_21_ci_
                currentConstrainedOut = d_22_cc_
                d_1_steps_ = (d_1_steps_) + (d_19_closeBudget_)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

