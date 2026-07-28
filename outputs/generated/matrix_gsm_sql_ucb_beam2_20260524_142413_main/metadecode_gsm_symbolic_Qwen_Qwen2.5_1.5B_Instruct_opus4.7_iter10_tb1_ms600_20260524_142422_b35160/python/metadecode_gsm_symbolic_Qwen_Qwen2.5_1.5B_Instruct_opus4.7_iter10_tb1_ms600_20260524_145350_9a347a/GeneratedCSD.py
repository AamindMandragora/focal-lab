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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. For EACH arithmetic step, write the calculation between << and >> markers like this: <<5+3=8>> or <<6*4=24>>. Each << >> contains exactly one short operation: numbers and operator, then '=', then the numeric result, and immediately close with >>. ALWAYS close every << with a matching >>. After all calculations, write '#### ' followed by the final numeric answer.\n\nExample: 'She had 5 cookies and ate 2 leaving <<5-2=3>>. Then she got 4 more: <<3+4=7>>. #### 7'")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_tokensSinceSpanEvent_: int
        d_2_tokensSinceSpanEvent_ = 0
        d_3_forceOpenAfter_: int
        d_3_forceOpenAfter_ = 40
        d_4_forcedOpensRemaining_: int
        d_4_forcedOpensRemaining_ = 2
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        if ((d_4_forcedOpensRemaining_) > (0)) and ((d_2_tokensSinceSpanEvent_) >= (d_3_forceOpenAfter_)):
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
                            d_1_steps_ = (d_1_steps_) + (1)
                            d_2_tokensSinceSpanEvent_ = 0
                            d_4_forcedOpensRemaining_ = (d_4_forcedOpensRemaining_) - (1)
                        elif True:
                            d_8_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                            d_8_next_ = out3_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_8_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_8_next_]))
                                if (d_8_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                    insideConstrainedOut = True
                                    currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                    d_2_tokensSinceSpanEvent_ = 0
                                elif True:
                                    d_2_tokensSinceSpanEvent_ = (d_2_tokensSinceSpanEvent_) + (1)
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
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_tokensSinceSpanEvent_ = 0
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_spanLen_: int
                        d_13_spanLen_ = len(currentConstrainedOut)
                        d_14_equalsCount_: int
                        out7_: int
                        out7_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_14_equalsCount_ = out7_
                        d_15_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if ((d_14_equalsCount_) >= (1)) and ((d_13_spanLen_) >= (4)):
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('2e1'), eosToken)
                            d_15_next_ = out8_
                        elif ((d_14_equalsCount_) >= (1)) and ((d_13_spanLen_) >= (2)):
                            out9_: _dafny.Seq
                            out9_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                            d_15_next_ = out9_
                        elif ((d_14_equalsCount_) == (0)) and ((d_13_spanLen_) >= (5)):
                            out10_: _dafny.Seq
                            out10_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('15e0'), eosToken)
                            d_15_next_ = out10_
                        elif ((d_14_equalsCount_) == (0)) and ((d_13_spanLen_) >= (2)):
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('4e0'), eosToken)
                            d_15_next_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_15_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_16_appG_: _dafny.Seq
                            d_17_appI_: bool
                            d_18_appC_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                            d_16_appG_ = out13_
                            d_17_appI_ = out14_
                            d_18_appC_ = out15_
                            generated = d_16_appG_
                            insideConstrainedOut = d_17_appI_
                            currentConstrainedOut = d_18_appC_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

