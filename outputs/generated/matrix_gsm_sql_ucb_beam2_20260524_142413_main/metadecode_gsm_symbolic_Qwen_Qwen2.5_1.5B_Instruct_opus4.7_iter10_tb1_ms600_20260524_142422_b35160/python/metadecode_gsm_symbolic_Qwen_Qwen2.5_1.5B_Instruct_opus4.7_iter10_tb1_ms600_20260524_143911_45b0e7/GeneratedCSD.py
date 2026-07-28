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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. Wrap every arithmetic calculation inside << >> markers, like <<5+3=8>>. For example: 'There are <<5+3=8>> apples, so <<8*2=16>> total.' After your reasoning, write #### followed by the final numeric answer.")))
        d_1_safeMaxSteps_: int
        if (maxSteps) >= (6):
            d_1_safeMaxSteps_ = (maxSteps) - (5)
        elif True:
            d_1_safeMaxSteps_ = maxSteps
        d_2_steps_: int
        d_2_steps_ = 0
        d_3_tokensSinceSpanEnd_: int
        d_3_tokensSinceSpanEnd_ = 0
        d_4_forceOpenAfter_: int
        d_4_forceOpenAfter_ = 22
        with _dafny.label("0"):
            while (d_2_steps_) < (d_1_safeMaxSteps_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if (d_3_tokensSinceSpanEnd_) >= (d_4_forceOpenAfter_):
                            d_5_openedG_: _dafny.Seq
                            d_6_openedI_: bool
                            d_7_openedC_: _dafny.Seq
                            out0_: _dafny.Seq
                            out1_: bool
                            out2_: _dafny.Seq
                            out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                            d_5_openedG_ = out0_
                            d_6_openedI_ = out1_
                            d_7_openedC_ = out2_
                            generated = d_5_openedG_
                            insideConstrainedOut = d_6_openedI_
                            currentConstrainedOut = d_7_openedC_
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_tokensSinceSpanEnd_ = 0
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_2_steps_ = (d_2_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_3_tokensSinceSpanEnd_ = 0
                                elif True:
                                    d_3_tokensSinceSpanEnd_ = (d_3_tokensSinceSpanEnd_) + (1)
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_closedG_: _dafny.Seq
                        d_10_closedI_: bool
                        d_11_closedC_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_closedG_ = out4_
                        d_10_closedI_ = out5_
                        d_11_closedC_ = out6_
                        generated = d_9_closedG_
                        insideConstrainedOut = d_10_closedI_
                        currentConstrainedOut = d_11_closedC_
                        d_2_steps_ = (d_2_steps_) + (1)
                        d_3_tokensSinceSpanEnd_ = 0
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                        d_13_next_ = out7_
                        d_2_steps_ = (d_2_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_14_appG_: _dafny.Seq
                            d_15_appI_: bool
                            d_16_appC_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_14_appG_ = out8_
                            d_15_appI_ = out9_
                            d_16_appC_ = out10_
                            generated = d_14_appG_
                            insideConstrainedOut = d_15_appI_
                            currentConstrainedOut = d_16_appC_
                    pass
            pass
        cost = d_2_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

