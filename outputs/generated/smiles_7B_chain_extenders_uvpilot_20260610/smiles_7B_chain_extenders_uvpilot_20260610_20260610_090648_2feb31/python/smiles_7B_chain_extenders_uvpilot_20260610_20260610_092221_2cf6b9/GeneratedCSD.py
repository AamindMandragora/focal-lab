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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate exactly one SMILES string for a chain extender molecule used in polyurethane synthesis. Chain extenders are small bifunctional molecules with at least 2 functional groups (hydroxyl -OH or amine -NH2). Examples: OCCO, OCCCCO, OCCCO, NCCN, NCCCCN, NCCO, OCCCCCO, NCCCCCCN, OCC(O)CO. Do NOT output single atoms. Output ONLY the SMILES string with multiple atoms.")))
        if (d_1_steps_) < (maxSteps):
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
                    if insideConstrainedOut:
                        if (parser).IsCompletePrefix(currentConstrainedOut):
                            d_5_cg_: _dafny.Seq
                            d_6_ci_: bool
                            d_7_cc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_5_cg_ = out3_
                            d_6_ci_ = out4_
                            d_7_cc_ = out5_
                            generated = d_5_cg_
                            insideConstrainedOut = d_6_ci_
                            currentConstrainedOut = d_7_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                            raise _dafny.Break("0")
                        elif True:
                            if ((d_1_steps_) + (2)) >= (maxSteps):
                                d_8_rg_: _dafny.Seq
                                d_9_rc_: _dafny.Seq
                                out6_: _dafny.Seq
                                out7_: _dafny.Seq
                                out6_, out7_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_8_rg_ = out6_
                                d_9_rc_ = out7_
                                generated = d_8_rg_
                                currentConstrainedOut = d_9_rc_
                                if ((d_1_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_10_cg_: _dafny.Seq
                                    d_11_ci_: bool
                                    d_12_cc_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_10_cg_ = out8_
                                    d_11_ci_ = out9_
                                    d_12_cc_ = out10_
                                    generated = d_10_cg_
                                    insideConstrainedOut = d_11_ci_
                                    currentConstrainedOut = d_12_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            d_13_constrainedPrompt_: _dafny.Seq
                            d_13_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_14_next_: _dafny.Seq
                            out11_: _dafny.Seq
                            out11_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_13_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('6e0'), 50, eosToken)
                            d_14_next_ = out11_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_14_next_) == (eosToken):
                                d_15_rg_: _dafny.Seq
                                d_16_rc_: _dafny.Seq
                                out12_: _dafny.Seq
                                out13_: _dafny.Seq
                                out12_, out13_ = (d_0_helpers_).RollbackConstrainedToComplete(parser, generated, currentConstrainedOut)
                                d_15_rg_ = out12_
                                d_16_rc_ = out13_
                                generated = d_15_rg_
                                currentConstrainedOut = d_16_rc_
                                if ((d_1_steps_) < (maxSteps)) and ((parser).IsCompletePrefix(currentConstrainedOut)):
                                    d_17_cg_: _dafny.Seq
                                    d_18_ci_: bool
                                    d_19_cc_: _dafny.Seq
                                    out14_: _dafny.Seq
                                    out15_: bool
                                    out16_: _dafny.Seq
                                    out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_17_cg_ = out14_
                                    d_18_ci_ = out15_
                                    d_19_cc_ = out16_
                                    generated = d_17_cg_
                                    insideConstrainedOut = d_18_ci_
                                    currentConstrainedOut = d_19_cc_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("0")
                            elif True:
                                d_20_isComplete_: bool
                                d_20_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if not(d_20_isComplete_):
                                    d_21_ag_: _dafny.Seq
                                    d_22_ai_: bool
                                    d_23_ac_: _dafny.Seq
                                    out17_: _dafny.Seq
                                    out18_: bool
                                    out19_: _dafny.Seq
                                    out17_, out18_, out19_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                    d_21_ag_ = out17_
                                    d_22_ai_ = out18_
                                    d_23_ac_ = out19_
                                    generated = d_21_ag_
                                    insideConstrainedOut = d_22_ai_
                                    currentConstrainedOut = d_23_ac_
                    elif True:
                        d_24_next_: _dafny.Seq
                        out20_: _dafny.Seq
                        out20_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_24_next_ = out20_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_24_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_24_next_]))
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

