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
        generated = generatedPrefix
        insideConstrainedOut = insideConstrained
        currentConstrainedOut = currentConstrained
        cost = 0
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_2_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_3_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_3_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_3_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_3_next_]))
                            if (d_3_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_4_isComplete_: bool
                        d_4_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_4_isComplete_:
                            d_5_closedGenerated_: _dafny.Seq
                            d_6_closedInside_: bool
                            d_7_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_closedGenerated_ = out2_
                            d_6_closedInside_ = out3_
                            d_7_closedCurrent_ = out4_
                            generated = d_5_closedGenerated_
                            insideConstrainedOut = d_6_closedInside_
                            currentConstrainedOut = d_7_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_8_stablePrefix_: _dafny.Seq
                            d_8_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (d_8_stablePrefix_)
                            (lm).GenerateLogits((d_9_constrainedPrompt_) + (currentConstrainedOut))
                            if (len(d_2_flatGroups_)) > (0):
                                d_10_prevTok_: _dafny.Seq
                                d_11_foundPrev_: bool
                                out5_: _dafny.Seq
                                out6_: bool
                                out5_, out6_ = (d_0_helpers_).LastTokenBefore(generated, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                                d_10_prevTok_ = out5_
                                d_11_foundPrev_ = out6_
                                if d_11_foundPrev_:
                                    d_12_activeIdx_: int
                                    out7_: int
                                    out7_ = VerifiedDecoderAgent.CSDHelpers.GroupContaining(validTokenGroups, d_10_prevTok_)
                                    d_12_activeIdx_ = out7_
                                    if (0) <= (d_12_activeIdx_):
                                        d_13_activeGroup_: _dafny.Seq
                                        d_13_activeGroup_ = (validTokenGroups)[d_12_activeIdx_]
                                        def lambda0_(forall_var_0_):
                                            d_14_t_: _dafny.Seq = forall_var_0_
                                            return not ((d_14_t_) in (d_13_activeGroup_)) or ((d_14_t_) in ((lm).Tokens))

                                        if _dafny.quantifier((d_13_activeGroup_).UniqueElements, True, lambda0_):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_13_activeGroup_, _dafny.BigRational('8e0'))
                                        d_15_otherPreferred_: _dafny.Seq
                                        out8_: _dafny.Seq
                                        out8_ = VerifiedDecoderAgent.CSDHelpers.SubtractTokenSets(d_2_flatGroups_, d_13_activeGroup_)
                                        d_15_otherPreferred_ = out8_
                                        if (len(d_15_otherPreferred_)) > (0):
                                            def lambda1_(forall_var_1_):
                                                d_16_t_: _dafny.Seq = forall_var_1_
                                                return not ((d_16_t_) in (d_15_otherPreferred_)) or ((d_16_t_) in ((lm).Tokens))

                                            if _dafny.quantifier((d_15_otherPreferred_).UniqueElements, True, lambda1_):
                                                (d_0_helpers_).PenalizeTokenLogits(lm, d_15_otherPreferred_, _dafny.BigRational('2e0'))
                                    elif True:
                                        def lambda2_(forall_var_2_):
                                            d_17_t_: _dafny.Seq = forall_var_2_
                                            return not ((d_17_t_) in (d_2_flatGroups_)) or ((d_17_t_) in ((lm).Tokens))

                                        if _dafny.quantifier((d_2_flatGroups_).UniqueElements, True, lambda2_):
                                            (d_0_helpers_).BoostTokenLogits(lm, d_2_flatGroups_, _dafny.BigRational('3e0'))
                                elif True:
                                    def lambda3_(forall_var_3_):
                                        d_18_t_: _dafny.Seq = forall_var_3_
                                        return not ((d_18_t_) in (d_2_flatGroups_)) or ((d_18_t_) in ((lm).Tokens))

                                    if _dafny.quantifier((d_2_flatGroups_).UniqueElements, True, lambda3_):
                                        (d_0_helpers_).BoostTokenLogits(lm, d_2_flatGroups_, _dafny.BigRational('3e0'))
                            (lm).MaskValidNextAndEos(parser, currentConstrainedOut, eosToken)
                            d_19_next_: _dafny.Seq
                            out9_: _dafny.Seq
                            out9_ = (lm).ChooseNextToken()
                            d_19_next_ = out9_
                            (d_0_helpers_).cost = (d_0_helpers_.cost) + (1)
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_19_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_20_appendedGenerated_: _dafny.Seq
                                d_21_appendedInside_: bool
                                d_22_appendedCurrent_: _dafny.Seq
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_19_next_)
                                d_20_appendedGenerated_ = out10_
                                d_21_appendedInside_ = out11_
                                d_22_appendedCurrent_ = out12_
                                generated = d_20_appendedGenerated_
                                insideConstrainedOut = d_21_appendedInside_
                                currentConstrainedOut = d_22_appendedCurrent_
                    pass
            pass
        cost = d_1_steps_
        if ((maxSteps) > (0)) and ((cost) == (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

