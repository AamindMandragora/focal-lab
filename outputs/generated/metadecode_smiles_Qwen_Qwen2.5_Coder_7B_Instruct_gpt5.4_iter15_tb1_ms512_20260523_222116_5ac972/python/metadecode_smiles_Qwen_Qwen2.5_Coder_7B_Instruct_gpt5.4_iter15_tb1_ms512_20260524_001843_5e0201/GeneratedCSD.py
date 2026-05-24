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
        d_3_preludeLimit_ = 8
        d_4_forcedOpenDone_: bool
        d_4_forcedOpenDone_ = insideConstrained
        d_5_rollbackLimit_: int
        d_5_rollbackLimit_ = 24
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (not(d_4_forcedOpenDone_)) and (((len(generated)) - (len(generatedPrefix))) >= (d_3_preludeLimit_)):
                            d_6_openedGenerated_: _dafny.Seq
                            d_7_openedInside_: bool
                            d_8_openedCurrent_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_6_openedGenerated_ = out0_
                            d_7_openedInside_ = out1_
                            d_8_openedCurrent_ = out2_
                            generated = d_6_openedGenerated_
                            insideConstrainedOut = d_7_openedInside_
                            currentConstrainedOut = d_8_openedCurrent_
                            d_4_forcedOpenDone_ = True
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_9_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_next_]))
                                if (d_9_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    d_10_enteredGenerated_: _dafny.Seq
                                    d_11_enteredInside_: bool
                                    d_12_enteredCurrent_: _dafny.Seq
                                    out4_: _dafny.Seq
                                    out5_: bool
                                    out6_: _dafny.Seq
                                    out4_, out5_, out6_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                    d_10_enteredGenerated_ = out4_
                                    d_11_enteredInside_ = out5_
                                    d_12_enteredCurrent_ = out6_
                                    generated = d_10_enteredGenerated_
                                    insideConstrainedOut = d_11_enteredInside_
                                    currentConstrainedOut = d_12_enteredCurrent_
                                    d_4_forcedOpenDone_ = True
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_13_closedGenerated_: _dafny.Seq
                        d_14_closedInside_: bool
                        d_15_closedCurrent_: _dafny.Seq
                        out7_: _dafny.Seq
                        out8_: bool
                        out9_: _dafny.Seq
                        out7_, out8_, out9_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_13_closedGenerated_ = out7_
                        d_14_closedInside_ = out8_
                        d_15_closedCurrent_ = out9_
                        generated = d_13_closedGenerated_
                        insideConstrainedOut = d_14_closedInside_
                        currentConstrainedOut = d_15_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif (len(currentConstrainedOut)) >= (d_5_rollbackLimit_):
                        d_16_rolledGenerated_: _dafny.Seq
                        d_17_rolledCurrent_: _dafny.Seq
                        out10_: _dafny.Seq
                        out11_: _dafny.Seq
                        out10_, out11_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_16_rolledGenerated_ = out10_
                        d_17_rolledCurrent_ = out11_
                        generated = d_16_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_17_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_18_stablePrefix_: _dafny.Seq
                        d_18_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_19_constrainedPrompt_: _dafny.Seq
                        d_19_constrainedPrompt_ = (prompt) + (d_18_stablePrefix_)
                        d_20_repeatedRecently_: bool
                        d_20_repeatedRecently_ = False
                        if (len(currentConstrainedOut)) > (0):
                            d_21_lastTok_: _dafny.Seq
                            d_21_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                            d_22_occ_: int = int(0)
                            out12_: int
                            out12_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, d_21_lastTok_)
                            d_22_occ_ = out12_
                            if (d_22_occ_) >= (d_2_repeatThreshold_):
                                d_20_repeatedRecently_ = True
                        d_23_nextIn_: _dafny.Seq
                        d_23_nextIn_ = eosToken
                        if d_20_repeatedRecently_:
                            out13_: _dafny.Seq
                            out13_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_23_nextIn_ = out13_
                        elif True:
                            d_24_gatedNext_: _dafny.Seq
                            d_25_wasConstrained_: bool
                            out14_: _dafny.Seq
                            out15_: bool
                            out14_, out15_ = (d_0_helpers_).ConfidenceGatedStep(lm, parser, d_19_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_24_gatedNext_ = out14_
                            d_25_wasConstrained_ = out15_
                            d_23_nextIn_ = d_24_gatedNext_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_23_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_26_appendedGenerated_: _dafny.Seq
                            d_27_appendedInside_: bool
                            d_28_appendedCurrent_: _dafny.Seq
                            out16_: _dafny.Seq
                            out17_: bool
                            out18_: _dafny.Seq
                            out16_, out17_, out18_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_23_nextIn_)
                            d_26_appendedGenerated_ = out16_
                            d_27_appendedInside_ = out17_
                            d_28_appendedCurrent_ = out18_
                            generated = d_26_appendedGenerated_
                            insideConstrainedOut = d_27_appendedInside_
                            currentConstrainedOut = d_28_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

