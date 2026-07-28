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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Solve this math word problem step by step. For each arithmetic operation, write the calculation between << and >> markers, e.g. <<5+3=8>> or <<6*4=24>>. Each span MUST contain exactly one short operation: numbers and operator, then '=', then the numeric result, then close immediately with >>. ALWAYS pair every << with a matching >>. After the calculations, finish with '#### <number>' giving the final numeric answer.\n\nExample: She had 5 cookies and ate 2 leaving <<5-2=3>>. Then she got 4 more <<3+4=7>>. #### 7")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_effectiveCap_: int
        if (maxSteps) > (220):
            d_2_effectiveCap_ = 220
        elif True:
            d_2_effectiveCap_ = maxSteps
        d_3_prevTok_: _dafny.Seq
        d_3_prevTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
        d_4_repeatCount_: int
        d_4_repeatCount_ = 0
        d_5_spanLen_: int
        d_5_spanLen_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (d_2_effectiveCap_):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_6_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_6_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_6_next_) == (eosToken):
                            raise _dafny.Break("0")
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_next_]))
                        if (d_6_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                            insideConstrainedOut = True
                            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                            d_5_spanLen_ = 0
                            d_3_prevTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            d_4_repeatCount_ = 0
                        elif True:
                            if (d_6_next_) == (d_3_prevTok_):
                                d_4_repeatCount_ = (d_4_repeatCount_) + (1)
                                if (d_4_repeatCount_) >= (6):
                                    raise _dafny.Break("0")
                            elif True:
                                d_3_prevTok_ = d_6_next_
                                d_4_repeatCount_ = 1
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_7_closedG_: _dafny.Seq
                        d_8_closedI_: bool
                        d_9_closedC_: _dafny.Seq
                        out1_: _dafny.Seq
                        out2_: bool
                        out3_: _dafny.Seq
                        out1_, out2_, out3_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_7_closedG_ = out1_
                        d_8_closedI_ = out2_
                        d_9_closedC_ = out3_
                        generated = d_7_closedG_
                        insideConstrainedOut = d_8_closedI_
                        currentConstrainedOut = d_9_closedC_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_5_spanLen_ = 0
                        d_3_prevTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_4_repeatCount_ = 0
                    elif ((d_5_spanLen_) >= (14)) or ((d_4_repeatCount_) >= (3)):
                        generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
                        insideConstrainedOut = False
                        currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_5_spanLen_ = 0
                        d_3_prevTok_ = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        d_4_repeatCount_ = 0
                    elif True:
                        d_10_constrainedPrompt_: _dafny.Seq
                        d_10_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_11_equalsCount_: int
                        out4_: int
                        out4_ = VerifiedDecoderAgent.CSDHelpers.CountTokenOccurrences(currentConstrainedOut, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "=")))
                        d_11_equalsCount_ = out4_
                        d_12_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                        if ((d_11_equalsCount_) >= (1)) and ((d_5_spanLen_) >= (3)):
                            out5_: _dafny.Seq
                            out5_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('2e1'), eosToken)
                            d_12_next_ = out5_
                        elif (d_11_equalsCount_) >= (1):
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "0")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "1")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "2")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "3")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "4")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "5")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "6")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "7")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "8")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "9")), _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]), _dafny.BigRational('6e0'), eosToken)
                            d_12_next_ = out6_
                        elif (d_5_spanLen_) >= (5):
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).SafeBoostedConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, _dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "="))]), _dafny.BigRational('12e0'), eosToken)
                            d_12_next_ = out7_
                        elif True:
                            out8_: _dafny.Seq
                            out8_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_10_constrainedPrompt_, currentConstrainedOut, eosToken)
                            d_12_next_ = out8_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_12_next_) == (eosToken):
                            raise _dafny.Break("0")
                        if (d_12_next_) == (d_3_prevTok_):
                            d_4_repeatCount_ = (d_4_repeatCount_) + (1)
                        elif True:
                            d_3_prevTok_ = d_12_next_
                            d_4_repeatCount_ = 1
                        d_13_appG_: _dafny.Seq
                        d_14_appI_: bool
                        d_15_appC_: _dafny.Seq
                        out9_: _dafny.Seq
                        out10_: bool
                        out11_: _dafny.Seq
                        out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_12_next_)
                        d_13_appG_ = out9_
                        d_14_appI_ = out10_
                        d_15_appC_ = out11_
                        generated = d_13_appG_
                        insideConstrainedOut = d_14_appI_
                        currentConstrainedOut = d_15_appC_
                        d_5_spanLen_ = (d_5_spanLen_) + (1)
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            generated = (generated) + (_dafny.SeqWithoutIsStrInference([_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ">>"))]))
            insideConstrainedOut = False
            currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
            d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

