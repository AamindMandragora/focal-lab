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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one novel SMILES string for an isocyanate molecule. Isocyanates contain the functional group N=C=O. Output only the SMILES string.")))
        d_1_steps_: int
        d_1_steps_ = 0
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_2_og_: _dafny.Seq
                        d_3_oi_: bool
                        d_4_oc_: _dafny.Seq
                        out0_: _dafny.Seq
                        out1_: bool
                        out2_: _dafny.Seq
                        out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                        d_2_og_ = out0_
                        d_3_oi_ = out1_
                        d_4_oc_ = out2_
                        generated = d_2_og_
                        insideConstrainedOut = d_3_oi_
                        currentConstrainedOut = d_4_oc_
                        d_1_steps_ = (d_1_steps_) + (1)
                    elif True:
                        d_5_stableLen_: int
                        d_5_stableLen_ = (len(generated)) - (len(currentConstrainedOut))
                        d_6_constrainedPrompt_: _dafny.Seq
                        d_6_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:d_5_stableLen_:]))
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_7_cg2_: _dafny.Seq
                            d_8_ci2_: bool
                            d_9_cc2_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_7_cg2_ = out3_
                            d_8_ci2_ = out4_
                            d_9_cc2_ = out5_
                            generated = d_7_cg2_
                            insideConstrainedOut = d_8_ci2_
                            currentConstrainedOut = d_9_cc2_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        if ((d_1_steps_) + (2)) >= (maxSteps):
                            d_10_rg_: _dafny.Seq
                            d_11_rc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: _dafny.Seq
                            out6_, out7_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_10_rg_ = out6_
                            d_11_rc_ = out7_
                            generated = d_10_rg_
                            currentConstrainedOut = d_11_rc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_12_cg2_: _dafny.Seq
                                d_13_ci2_: bool
                                d_14_cc2_: _dafny.Seq
                                out8_: _dafny.Seq
                                out9_: bool
                                out10_: _dafny.Seq
                                out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_12_cg2_ = out8_
                                d_13_ci2_ = out9_
                                d_14_cc2_ = out10_
                                generated = d_12_cg2_
                                insideConstrainedOut = d_13_ci2_
                                currentConstrainedOut = d_14_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        d_15_next_: _dafny.Seq
                        out11_: _dafny.Seq
                        out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_6_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 20, eosToken)
                        d_15_next_ = out11_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_15_next_) == (eosToken):
                            d_16_rg_: _dafny.Seq
                            d_17_rc_: _dafny.Seq
                            out12_: _dafny.Seq
                            out13_: _dafny.Seq
                            out12_, out13_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                            d_16_rg_ = out12_
                            d_17_rc_ = out13_
                            generated = d_16_rg_
                            currentConstrainedOut = d_17_rc_
                            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                d_18_cg2_: _dafny.Seq
                                d_19_ci2_: bool
                                d_20_cc2_: _dafny.Seq
                                out14_: _dafny.Seq
                                out15_: bool
                                out16_: _dafny.Seq
                                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                d_18_cg2_ = out14_
                                d_19_ci2_ = out15_
                                d_20_cc2_ = out16_
                                generated = d_18_cg2_
                                insideConstrainedOut = d_19_ci2_
                                currentConstrainedOut = d_20_cc2_
                                d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            d_21_isComplete_: bool
                            d_21_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                            if d_21_isComplete_:
                                if (d_1_steps_) < (maxSteps):
                                    d_22_cg2_: _dafny.Seq
                                    d_23_ci2_: bool
                                    d_24_cc2_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_22_cg2_ = out17_
                                    d_23_ci2_ = out18_
                                    d_24_cc2_ = out19_
                                    generated = d_22_cg2_
                                    insideConstrainedOut = d_23_ci2_
                                    currentConstrainedOut = d_24_cc2_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_25_ag_: _dafny.Seq
                                d_26_ai_: bool
                                d_27_ac_: _dafny.Seq
                                out20_: _dafny.Seq
                                out21_: bool
                                out22_: _dafny.Seq
                                out20_, out21_, out22_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_15_next_)
                                d_25_ag_ = out20_
                                d_26_ai_ = out21_
                                d_27_ac_ = out22_
                                generated = d_25_ag_
                                insideConstrainedOut = d_26_ai_
                                currentConstrainedOut = d_27_ac_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

