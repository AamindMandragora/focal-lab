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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a valid SMILES string for the requested molecular class. Prefer one complete <<SMILES>> span; keep the content inside the span as a single valid SMILES, maintain parser-valid prefixes, and close the span immediately once the molecule is complete.")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_repeatThreshold_: int
        d_2_repeatThreshold_ = 2
        d_3_preludeLimit_: int
        d_3_preludeLimit_ = 10
        d_4_forcedOpenDone_: bool
        d_4_forcedOpenDone_ = insideConstrained
        d_5_rollbackLimit_: int
        d_5_rollbackLimit_ = 24
        d_6_narrowThreshold_: int
        d_6_narrowThreshold_ = 12
        d_7_cautionLength_: int
        d_7_cautionLength_ = 8
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_4_forcedOpenDone_)) and (((len(generated)) - (len(generatedPrefix))) >= (d_3_preludeLimit_)):
                            d_8_openedGenerated_: _dafny.Seq
                            d_9_openedInside_: bool
                            d_10_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_8_openedGenerated_ = out0_
                            d_9_openedInside_ = out1_
                            d_10_openedCurrent_ = out2_
                            generated = d_8_openedGenerated_
                            insideConstrainedOut = d_9_openedInside_
                            currentConstrainedOut = d_10_openedCurrent_
                            d_4_forcedOpenDone_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_11_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_11_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_11_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_11_next_]))
                                if (d_11_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_12_enteredGenerated_: _dafny.Seq
                                    d_13_enteredInside_: bool
                                    d_14_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_12_enteredGenerated_ = out4_
                                    d_13_enteredInside_ = out5_
                                    d_14_enteredCurrent_ = out6_
                                    generated = d_12_enteredGenerated_
                                    insideConstrainedOut = d_13_enteredInside_
                                    currentConstrainedOut = d_14_enteredCurrent_
                                    d_4_forcedOpenDone_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_15_closedGenerated_: _dafny.Seq
                        d_16_closedInside_: bool
                        d_17_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_15_closedGenerated_ = out7_
                        d_16_closedInside_ = out8_
                        d_17_closedCurrent_ = out9_
                        generated = d_15_closedGenerated_
                        insideConstrainedOut = d_16_closedInside_
                        currentConstrainedOut = d_17_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_5_rollbackLimit_):
                        d_18_rolledGenerated_: _dafny.Seq
                        d_19_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_18_rolledGenerated_ = out10_
                        d_19_rolledCurrent_ = out11_
                        generated = d_18_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_19_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_20_stablePrefix_: _dafny.Seq
                        d_20_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_21_constrainedPrompt_: _dafny.Seq
                        d_21_constrainedPrompt_ = (prompt) + (d_20_stablePrefix_)
                        d_22_repeatedRecently_: bool
                        d_22_repeatedRecently_ = False
                        if (len(currentConstrainedOut)) > (0):
                            d_23_lastTok_: _dafny.Seq
                            d_23_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                            d_24_occ_: int = int(0)
                            out12_: int
                            out12_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, d_23_lastTok_)
                            d_24_occ_ = out12_
                            if (d_24_occ_) >= (d_2_repeatThreshold_):
                                d_22_repeatedRecently_ = True
                        d_25_validCount_: int
                        out13_: int
                        out13_ = (d_0_helpers_).ValidTokenCount(parser, currentConstrainedOut)
                        d_25_validCount_ = out13_
                        d_26_nextIn_: _dafny.Seq
                        d_26_nextIn_ = eosToken
                        if d_22_repeatedRecently_:
                            out14_: _dafny.Seq
                            out14_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_26_nextIn_ = out14_
                        elif ((d_25_validCount_) <= (d_6_narrowThreshold_)) or ((len(currentConstrainedOut)) >= (d_7_cautionLength_)):
                            out15_: _dafny.Seq
                            out15_ = (d_0_helpers_).SafePenalizedConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))]), _dafny.BigRational('4e0'), eosToken)
                            d_26_nextIn_ = out15_
                        elif True:
                            out16_: _dafny.Seq
                            out16_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_21_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_26_nextIn_ = out16_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_26_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_27_appendedGenerated_: _dafny.Seq
                            d_28_appendedInside_: bool
                            d_29_appendedCurrent_: _dafny.Seq
                            out17_: _dafny.Seq
                            out18_: bool
                            out19_: _dafny.Seq
                            out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_26_nextIn_)
                            d_27_appendedGenerated_ = out17_
                            d_28_appendedInside_ = out18_
                            d_29_appendedCurrent_ = out19_
                            generated = d_27_appendedGenerated_
                            insideConstrainedOut = d_28_appendedInside_
                            currentConstrainedOut = d_29_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

