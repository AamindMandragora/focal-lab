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
        d_1_steps_: int
        d_1_steps_ = 0
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SMILES for a novel acrylate molecule. Acrylates contain C=CC(=O)O or C=C(C)C(=O)O. Generate a complete multi-atom SMILES like C=CC(=O)OCCC or C=C(C)C(=O)OCCCO or C=CC(=O)OCC(C)C. Output only the SMILES string.")))
        if ((d_1_steps_) < (maxSteps)) and (not(insideConstrainedOut)):
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
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        raise _dafny.Break("0")
                    d_5_constrainedPrompt_: _dafny.Seq
                    d_5_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                    d_6_next_: _dafny.Seq
                    out3_: _dafny.Seq
                    out3_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_5_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('2e0'), eosToken)
                    d_6_next_ = out3_
                    d_1_steps_ = (d_1_steps_) + (1)
                    if (d_6_next_) == (eosToken):
                        d_7_rg_: _dafny.Seq
                        d_8_rc_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: _dafny.Seq
                        out4_, out5_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                        d_7_rg_ = out4_
                        d_8_rc_ = out5_
                        generated = d_7_rg_
                        currentConstrainedOut = d_8_rc_
                        if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                            d_9_fg_: _dafny.Seq
                            d_10_fi_: bool
                            d_11_fc_: _dafny.Seq
                            out6_: _dafny.Seq
                            out7_: bool
                            out8_: _dafny.Seq
                            out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_9_fg_ = out6_
                            d_10_fi_ = out7_
                            d_11_fc_ = out8_
                            generated = d_9_fg_
                            insideConstrainedOut = d_10_fi_
                            currentConstrainedOut = d_11_fc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        raise _dafny.Break("0")
                    elif True:
                        d_12_isComplete_: bool
                        d_12_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        d_13_valid_: bool
                        out9_: bool
                        out9_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_6_next_)
                        d_13_valid_ = out9_
                        if (d_13_valid_) and (not(d_12_isComplete_)):
                            d_14_ag_: _dafny.Seq
                            d_15_ai_: bool
                            d_16_ac_: _dafny.Seq
                            out10_: _dafny.Seq
                            out11_: bool
                            out12_: _dafny.Seq
                            out10_, out11_, out12_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_6_next_)
                            d_14_ag_ = out10_
                            d_15_ai_ = out11_
                            d_16_ac_ = out12_
                            generated = d_14_ag_
                            insideConstrainedOut = d_15_ai_
                            currentConstrainedOut = d_16_ac_
                        if (d_1_steps_) < (maxSteps):
                            d_17_cg2_: _dafny.Seq
                            d_18_ci2_: bool
                            d_19_cc2_: _dafny.Seq
                            d_20_closed_: bool
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out16_: bool
                            out13_, out14_, out15_, out16_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                            d_17_cg2_ = out13_
                            d_18_ci2_ = out14_
                            d_19_cc2_ = out15_
                            d_20_closed_ = out16_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if d_20_closed_:
                                generated = d_17_cg2_
                                insideConstrainedOut = d_18_ci2_
                                currentConstrainedOut = d_19_cc2_
                                raise _dafny.Break("0")
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_21_rg_: _dafny.Seq
            d_22_rc_: _dafny.Seq
            out17_: _dafny.Seq
            out18_: _dafny.Seq
            out17_, out18_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
            d_21_rg_ = out17_
            d_22_rc_ = out18_
            generated = d_21_rg_
            currentConstrainedOut = d_22_rc_
            if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                d_23_fg_: _dafny.Seq
                d_24_fi_: bool
                d_25_fc_: _dafny.Seq
                out19_: _dafny.Seq
                out20_: bool
                out21_: _dafny.Seq
                out19_, out20_, out21_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_23_fg_ = out19_
                d_24_fi_ = out20_
                d_25_fc_ = out21_
                generated = d_23_fg_
                insideConstrainedOut = d_24_fi_
                currentConstrainedOut = d_25_fc_
                d_1_steps_ = (d_1_steps_) + (1)
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

