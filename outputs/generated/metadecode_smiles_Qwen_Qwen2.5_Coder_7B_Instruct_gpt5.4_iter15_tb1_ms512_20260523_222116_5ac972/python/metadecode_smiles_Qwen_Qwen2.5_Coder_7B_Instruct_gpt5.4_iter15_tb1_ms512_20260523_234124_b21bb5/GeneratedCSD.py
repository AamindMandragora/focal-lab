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
        insideConstrainedOut = True
        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
        cost = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Output only one valid SMILES string matching the requested molecular class. Do not add explanation or visible delimiters. Keep every prefix parser-valid and stop once one complete molecule has been formed.")))
        if insideConstrained:
            out0_: _dafny.Seq
            out0_ = VerifiedDecoderAgent.CSDHelpers.RollbackToValidPrefix(parser, currentConstrained)
            currentConstrainedOut = out0_
            generated = (_dafny.SeqWithoutIsStrInference((generatedPrefix)[:(len(generatedPrefix)) - (len(currentConstrained)):])) + (currentConstrainedOut)
        elif True:
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            generated = generatedPrefix
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_repeatThreshold_: int
        d_2_repeatThreshold_ = 2
        d_3_rollbackLimit_: int
        d_3_rollbackLimit_ = 40
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if (parser).IsCompletePrefix(currentConstrainedOut):
                        d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif (len(currentConstrainedOut)) >= (d_3_rollbackLimit_):
                        d_4_rolledGenerated_: _dafny.Seq
                        d_5_rolledCurrent_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: _dafny.Seq
                        out1_, out2_ = (d_0_helpers_).RollbackConstrainedSuffix(parser, generated, currentConstrainedOut)
                        d_4_rolledGenerated_ = out1_
                        d_5_rolledCurrent_ = out2_
                        generated = d_4_rolledGenerated_
                        insideConstrainedOut = True
                        currentConstrainedOut = d_5_rolledCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_6_stablePrefix_: _dafny.Seq
                        d_6_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                        d_7_constrainedPrompt_: _dafny.Seq
                        d_7_constrainedPrompt_ = (prompt) + (d_6_stablePrefix_)
                        d_8_repeatedRecently_: bool
                        d_8_repeatedRecently_ = False
                        if (len(currentConstrainedOut)) > (0):
                            d_9_lastTok_: _dafny.Seq
                            d_9_lastTok_ = (currentConstrainedOut)[(len(currentConstrainedOut)) - (1)]
                            d_10_occ_: int = int(0)
                            out3_: int
                            out3_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, d_9_lastTok_)
                            d_10_occ_ = out3_
                            if (d_10_occ_) >= (d_2_repeatThreshold_):
                                d_8_repeatedRecently_ = True
                        d_11_nextIn_: _dafny.Seq
                        d_11_nextIn_ = eosToken
                        if d_8_repeatedRecently_:
                            out4_: _dafny.Seq
                            out4_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                            d_11_nextIn_ = out4_
                        elif True:
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_7_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                            d_11_nextIn_ = out5_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_11_nextIn_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_12_appendedGenerated_: _dafny.Seq
                            d_13_appendedInside_: bool
                            d_14_appendedCurrent_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_11_nextIn_)
                            d_12_appendedGenerated_ = out6_
                            d_13_appendedInside_ = out7_
                            d_14_appendedCurrent_ = out8_
                            generated = d_12_appendedGenerated_
                            insideConstrainedOut = d_13_appendedInside_
                            currentConstrainedOut = d_14_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

