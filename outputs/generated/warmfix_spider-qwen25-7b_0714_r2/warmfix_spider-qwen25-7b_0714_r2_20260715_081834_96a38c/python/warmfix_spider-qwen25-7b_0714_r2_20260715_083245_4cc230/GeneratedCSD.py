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
        if (maxSteps) == (0):
            return generated, insideConstrainedOut, currentConstrainedOut, cost
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output exactly: SQL: <<your SQL query here>>. The SQL query must be complete and syntactically valid. Use only tables and columns from the schema.")))
        d_1_steps_: int
        d_1_steps_ = 0
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_2_prefixBudget_: int
            d_2_prefixBudget_ = 4
            if (d_2_prefixBudget_) > ((maxSteps) - (d_1_steps_)):
                d_2_prefixBudget_ = (maxSteps) - (d_1_steps_)
            d_3_chunkOut_: _dafny.Seq
            d_4_stoppedOnOpen_: bool
            d_5_stoppedOnEos_: bool
            d_6_chunkSteps_: int
            out0_: _dafny.Seq
            out1_: bool
            out2_: bool
            out3_: int
            out0_, out1_, out2_, out3_ = (d_0_helpers_).UnconstrainedChunk(lm, prompt, generated, d_2_prefixBudget_, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")), eosToken)
            d_3_chunkOut_ = out0_
            d_4_stoppedOnOpen_ = out1_
            d_5_stoppedOnEos_ = out2_
            d_6_chunkSteps_ = out3_
            d_1_steps_ = (d_1_steps_) + (d_6_chunkSteps_)
            generated = d_3_chunkOut_
            if d_4_stoppedOnOpen_:
                insideConstrainedOut = True
                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            elif d_5_stoppedOnEos_:
                cost = d_1_steps_
                return generated, insideConstrainedOut, currentConstrainedOut, cost
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
            d_7_openedGenerated_: _dafny.Seq
            d_8_openedInside_: bool
            d_9_openedCurrent_: _dafny.Seq
            out4_: _dafny.Seq
            out5_: bool
            out6_: _dafny.Seq
            out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
            d_7_openedGenerated_ = out4_
            d_8_openedInside_ = out5_
            d_9_openedCurrent_ = out6_
            generated = d_7_openedGenerated_
            insideConstrainedOut = d_8_openedInside_
            currentConstrainedOut = d_9_openedCurrent_
            d_1_steps_ = (d_1_steps_) + (1)
        with _dafny.label("0"):
            while (insideConstrainedOut) and (((d_1_steps_) + (2)) <= (maxSteps)):
                with _dafny.c_label("0"):
                    d_10_closeBudget_: int
                    d_10_closeBudget_ = 80
                    if ((d_1_steps_) + (d_10_closeBudget_)) >= (maxSteps):
                        raise _dafny.Break("0")
                    d_11_constrainedPrompt_: _dafny.Seq
                    d_11_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_12_next_: _dafny.Seq
                    out7_: _dafny.Seq
                    out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_11_constrainedPrompt_, currentConstrainedOut, eosToken)
                    d_12_next_ = out7_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_12_next_) == (eosToken):
                        raise _dafny.Break("0")
                    d_13_valid_: bool
                    out8_: bool
                    out8_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_next_)
                    d_13_valid_ = out8_
                    if d_13_valid_:
                        d_14_ng_: _dafny.Seq
                        d_15_ni_: bool
                        d_16_nc_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                        d_14_ng_ = out9_
                        d_15_ni_ = out10_
                        d_16_nc_ = out11_
                        generated = d_14_ng_
                        insideConstrainedOut = d_15_ni_
                        currentConstrainedOut = d_16_nc_
                        d_17_cg2_: _dafny.Seq
                        d_18_ci2_: bool
                        d_19_cc2_: _dafny.Seq
                        d_20_closed2_: bool
                        out12_: _dafny.Seq
                        out13_: bool
                        out14_: _dafny.Seq
                        out15_: bool
                        out12_, out13_, out14_, out15_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_17_cg2_ = out12_
                        d_18_ci2_ = out13_
                        d_19_cc2_ = out14_
                        d_20_closed2_ = out15_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_20_closed2_:
                            generated = d_17_cg2_
                            insideConstrainedOut = d_18_ci2_
                            currentConstrainedOut = d_19_cc2_
                            raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_21_closeBudget2_: int
            d_21_closeBudget2_ = (maxSteps) - (d_1_steps_)
            d_22_fg_: _dafny.Seq
            d_23_fi_: bool
            d_24_fc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_21_closeBudget2_)
            d_22_fg_ = out16_
            d_23_fi_ = out17_
            d_24_fc_ = out18_
            generated = d_22_fg_
            insideConstrainedOut = d_23_fi_
            currentConstrainedOut = d_24_fc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

