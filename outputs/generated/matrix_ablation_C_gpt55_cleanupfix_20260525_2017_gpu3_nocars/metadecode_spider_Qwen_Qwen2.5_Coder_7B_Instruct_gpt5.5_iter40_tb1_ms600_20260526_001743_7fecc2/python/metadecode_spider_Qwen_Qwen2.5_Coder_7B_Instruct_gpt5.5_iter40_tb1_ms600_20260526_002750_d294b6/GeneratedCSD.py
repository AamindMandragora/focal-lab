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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Return only one answer in exactly this format: SQL: <<SELECT ...>>. Use the schema and question to write a single valid SQLite query. Do not explain. Do not emit text after >>.")))
        if (maxSteps) == (0):
            cost = 0
        elif True:
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            if (maxSteps) == (1):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:"))]))
                cost = 1
            elif (maxSteps) == (2):
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
                cost = 2
            elif True:
                generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "SQL:")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, " "))]))
                d_1_openedGenerated_: _dafny.Seq
                d_2_openedInside_: bool
                d_3_openedCurrent_: _dafny.Seq
                out0_: _dafny.Seq
                out1_: bool
                out2_: _dafny.Seq
                out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                d_1_openedGenerated_ = out0_
                d_2_openedInside_ = out1_
                d_3_openedCurrent_ = out2_
                generated = d_1_openedGenerated_
                insideConstrainedOut = d_2_openedInside_
                currentConstrainedOut = d_3_openedCurrent_
                d_4_steps_: int
                d_4_steps_ = 3
                if (maxSteps) > (4):
                    d_5_firstCap_: int
                    d_5_firstCap_ = (maxSteps) - (4)
                    if (d_5_firstCap_) > (160):
                        d_5_firstCap_ = 160
                    d_6_constrainedPrompt_: _dafny.Seq
                    d_6_constrainedPrompt_ = (prompt) + (generated)
                    d_7_symbolGenerated_: _dafny.Seq
                    d_8_symbolOut_: _dafny.Seq
                    d_9_hitEos_: bool
                    d_10_used_: int
                    out3_: _dafny.Seq
                    out4_: _dafny.Seq
                    out5_: bool
                    out6_: int
                    out3_, out4_, out5_, out6_ = (d_0_helpers_).ConstrainedSymbolInGenerated(lm, parser, d_6_constrainedPrompt_, generated, currentConstrainedOut, d_5_firstCap_, eosToken)
                    d_7_symbolGenerated_ = out3_
                    d_8_symbolOut_ = out4_
                    d_9_hitEos_ = out5_
                    d_10_used_ = out6_
                    generated = d_7_symbolGenerated_
                    currentConstrainedOut = d_8_symbolOut_
                    insideConstrainedOut = True
                    d_4_steps_ = (d_4_steps_) + (d_10_used_)
                    if ((not((parser).IsCompletePrefix(currentConstrainedOut))) and (not(d_9_hitEos_))) and (((d_4_steps_) + (1)) < (maxSteps)):
                        d_11_rem_: int
                        d_11_rem_ = ((maxSteps) - (d_4_steps_)) - (1)
                        if (d_11_rem_) > (80):
                            d_11_rem_ = 80
                        if (d_11_rem_) > (0):
                            d_12_stablePrefix_: _dafny.Seq
                            d_12_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_13_repairPrompt_: _dafny.Seq
                            d_13_repairPrompt_ = (prompt) + (d_12_stablePrefix_)
                            d_14_rolledCurrent_: _dafny.Seq
                            d_15_rolledSteps_: int
                            d_16_rolledEos_: bool
                            out7_: _dafny.Seq
                            out8_: int
                            out9_: bool
                            out7_, out8_, out9_ = (d_0_helpers_).RolloutConstrainedWithPenalties(lm, parser, d_13_repairPrompt_, currentConstrainedOut, d_11_rem_, _dafny.SeqWithoutIsStrInference([eosToken]), _dafny.BigRational('1e1'), eosToken)
                            d_14_rolledCurrent_ = out7_
                            d_15_rolledSteps_ = out8_
                            d_16_rolledEos_ = out9_
                            generated = (d_12_stablePrefix_) + (d_14_rolledCurrent_)
                            currentConstrainedOut = d_14_rolledCurrent_
                            insideConstrainedOut = True
                            d_4_steps_ = (d_4_steps_) + (d_15_rolledSteps_)
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_4_steps_) < (maxSteps)):
                        d_17_closedGenerated_: _dafny.Seq
                        d_18_closedInside_: bool
                        d_19_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_17_closedGenerated_ = out10_
                        d_18_closedInside_ = out11_
                        d_19_closedCurrent_ = out12_
                        generated = d_17_closedGenerated_
                        insideConstrainedOut = d_18_closedInside_
                        currentConstrainedOut = d_19_closedCurrent_
                        d_4_steps_ = (d_4_steps_) + (1)
                cost = d_4_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

