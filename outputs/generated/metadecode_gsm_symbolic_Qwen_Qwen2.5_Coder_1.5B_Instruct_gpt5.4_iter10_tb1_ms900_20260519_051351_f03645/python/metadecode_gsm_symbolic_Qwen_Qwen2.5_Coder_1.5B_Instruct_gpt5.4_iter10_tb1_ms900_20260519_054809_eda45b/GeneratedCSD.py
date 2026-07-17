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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve the math word problem step by step. After the reasoning, output exactly one final numeric answer inside << >>, and keep only the final answer inside that span.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_answerSpanOpened_: bool
        d_2_answerSpanOpened_ = insideConstrained
        d_3_freePrefixCap_: int
        d_3_freePrefixCap_ = 96
        d_4_rollbackLimit_: int
        d_4_rollbackLimit_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_2_answerSpanOpened_)) and ((d_1_steps_) >= (d_3_freePrefixCap_)):
                            d_5_openedGenerated_: _dafny.Seq
                            d_6_openedInside_: bool
                            d_7_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedGenerated_ = out0_
                            d_6_openedInside_ = out1_
                            d_7_openedCurrent_ = out2_
                            generated = d_5_openedGenerated_
                            insideConstrainedOut = d_6_openedInside_
                            currentConstrainedOut = d_7_openedCurrent_
                            d_2_answerSpanOpened_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                if (not(d_2_answerSpanOpened_)) and ((d_1_steps_) < (maxSteps)):
                                    d_9_openedGenerated2_: _dafny.Seq
                                    d_10_openedInside2_: bool
                                    d_11_openedCurrent2_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                                    d_9_openedGenerated2_ = out4_
                                    d_10_openedInside2_ = out5_
                                    d_11_openedCurrent2_ = out6_
                                    generated = d_9_openedGenerated2_
                                    insideConstrainedOut = d_10_openedInside2_
                                    currentConstrainedOut = d_11_openedCurrent2_
                                    d_2_answerSpanOpened_ = True
                                    d_1_steps_ = (d_1_steps_) + (1)
                                elif True:
                                    raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_12_enteredGenerated_: _dafny.Seq
                                    d_13_enteredInside_: bool
                                    d_14_enteredCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_12_enteredGenerated_ = out7_
                                    d_13_enteredInside_ = out8_
                                    d_14_enteredCurrent_ = out9_
                                    generated = d_12_enteredGenerated_
                                    insideConstrainedOut = d_13_enteredInside_
                                    currentConstrainedOut = d_14_enteredCurrent_
                                    d_2_answerSpanOpened_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: bool
                        out12_: _dafny.Seq
                        out10_, out11_, out12_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out10_
                        d_16_closedInside_ = out11_
                        d_17_closedCurrent_ = out12_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (len(currentConstrainedOut)) >= (d_4_rollbackLimit_):
                        d_18_rolledGenerated_: _dafny.Seq
                        d_19_rolledCurrent_: _dafny.Seq
                        out13_: _dafny.Seq
                        out14_: _dafny.Seq
                        out13_, out14_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_18_rolledGenerated_ = out13_
                        d_19_rolledCurrent_ = out14_
                        generated = d_18_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_19_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_20_stablePrefix_: _dafny.Seq
                        d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                        d_22_nextIn_: _dafny.Seq
                        d_22_nextIn_ = eosToken
                        d_23_validCount_: int
                        out15_: int
                        out15_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_23_validCount_ = out15_
                        if (d_23_validCount_) <= (8):
                            d_24_candidates_: _dafny.Seq
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).TopValidCandidates(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, 4, eosToken)
                            d_24_candidates_ = out16_
                            d_22_nextIn_ = (d_24_candidates_)[0]
                            if ((d_22_nextIn_) == (eosToken)) and ((len(d_24_candidates_)) > (1)):
                                d_22_nextIn_ = (d_24_candidates_)[1]
                        elif (len(validTokenGroups)) > (0):
                            d_25_nextGroup_: _dafny.Seq
                            out17_: _dafny.Seq
                            out17_ = (d_0_helpers_).GroupBoostedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), eosToken)
                            d_25_nextGroup_ = out17_
                            d_22_nextIn_ = d_25_nextGroup_
                        elif True:
                            d_26_nextConf_: _dafny.Seq
                            d_27_wasConstrained_: bool
                            out18_: _dafny.Seq
                            out19_: bool
                            out18_, out19_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_26_nextConf_ = out18_
                            d_27_wasConstrained_ = out19_
                            d_22_nextIn_ = d_26_nextConf_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_22_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_28_appendedGenerated_: _dafny.Seq
                            d_29_appendedInside_: bool
                            d_30_appendedCurrent_: _dafny.Seq
                            out20_: _dafny.Seq
                            out21_: bool
                            out22_: _dafny.Seq
                            out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_22_nextIn_)
                            d_28_appendedGenerated_ = out20_
                            d_29_appendedInside_ = out21_
                            d_30_appendedCurrent_ = out22_
                            generated = d_28_appendedGenerated_
                            insideConstrainedOut = d_29_appendedInside_
                            currentConstrainedOut = d_30_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

