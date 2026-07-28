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
        out0_: _dafny.Seq
        out1_: bool
        out2_: _dafny.Seq
        out3_: int
        out0_, out1_, out2_, out3_ = default__.AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken)
        generated = out0_
        insideConstrainedOut = out1_
        currentConstrainedOut = out2_
        cost = out3_
        if ((maxSteps) > (0)) and ((cost) <= (0)):
            cost = 1
        return generated, insideConstrainedOut, currentConstrainedOut, cost

    @staticmethod
    def AuthorBody(lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained, maxSteps, stepTokenBudget, validTokenGroups, eosToken):
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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate ONE valid SMILES for a chain_extender molecule. A chain extender is a bifunctional molecule with exactly two reactive groups used in polyurethane/polyurea synthesis. Choose from LONGER chain extenders (more than 4 atoms): 1,4-butanediol OCCCCO, 1,6-hexanediol OCCCCCCO, 1,5-pentanediol OCCCCCO, diethylene glycol OCCOCCO, 2-methyl-1,3-propanediol OCC(C)CO, neopentyl glycol OCC(C)(C)CO, 1,3-propanediol OCCCО, 1,4-diaminobutane NCCCCN, 1,6-diaminohexane NCCCCCCN, 4-amino-1-butanol NCCCCO, 5-amino-1-pentanol NCCCCCO, 3-amino-1-propanol NCCCО, 2-amino-1-butanol NCC(O)CC. Output ONLY the SMILES string with NO explanation.")))
        if (not(insideConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
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
            while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                with _dafny.c_label("0"):
                    d_5_seqLen_: int
                    d_5_seqLen_ = len(currentConstrainedOut)
                    if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_5_seqLen_) >= (8)):
                        if (d_1_steps_) < (maxSteps):
                            d_6_cg_: _dafny.Seq
                            d_7_ci_: bool
                            d_8_cc_: _dafny.Seq
                            out3_: _dafny.Seq
                            out4_: bool
                            out5_: _dafny.Seq
                            out3_, out4_, out5_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_cg_ = out3_
                            d_7_ci_ = out4_
                            d_8_cc_ = out5_
                            generated = d_6_cg_
                            insideConstrainedOut = d_7_ci_
                            currentConstrainedOut = d_8_cc_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            raise _dafny.Break("0")
                    elif True:
                        d_9_constrainedPrompt_: _dafny.Seq
                        d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_10_next_: _dafny.Seq
                        d_10_next_ = eosToken
                        if (d_5_seqLen_) <= (2):
                            out6_: _dafny.Seq
                            out6_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('16e-1'), eosToken)
                            d_10_next_ = out6_
                        elif (d_5_seqLen_) <= (6):
                            if (_dafny.euclidian_modulus(d_5_seqLen_, 2)) == (0):
                                out7_: _dafny.Seq
                                out7_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('4e0'), eosToken)
                                d_10_next_ = out7_
                            elif True:
                                out8_: _dafny.Seq
                                out8_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('14e-1'), eosToken)
                                d_10_next_ = out8_
                        elif (d_5_seqLen_) <= (12):
                            if (_dafny.euclidian_modulus(d_5_seqLen_, 3)) == (0):
                                out9_: _dafny.Seq
                                out9_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('3e0'), eosToken)
                                d_10_next_ = out9_
                            elif (_dafny.euclidian_modulus(d_5_seqLen_, 3)) == (1):
                                out10_: _dafny.Seq
                                out10_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('13e-1'), eosToken)
                                d_10_next_ = out10_
                            elif True:
                                out11_: _dafny.Seq
                                out11_ = (d_0_helpers_).ConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, eosToken)
                                d_10_next_ = out11_
                        elif True:
                            out12_: _dafny.Seq
                            out12_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('1e0'), eosToken)
                            d_10_next_ = out12_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_10_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_11_ag_: _dafny.Seq
                            d_12_ai_: bool
                            d_13_ac_: _dafny.Seq
                            out13_: _dafny.Seq
                            out14_: bool
                            out15_: _dafny.Seq
                            out13_, out14_, out15_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                            d_11_ag_ = out13_
                            d_12_ai_ = out14_
                            d_13_ac_ = out15_
                            generated = d_11_ag_
                            insideConstrainedOut = d_12_ai_
                            currentConstrainedOut = d_13_ac_
                    pass
            pass
        if (insideConstrainedOut) and ((d_1_steps_) < (maxSteps)):
            d_14_closeBudget_: int
            d_14_closeBudget_ = (maxSteps) - (d_1_steps_)
            d_15_cg_: _dafny.Seq
            d_16_ci_: bool
            d_17_cc_: _dafny.Seq
            out16_: _dafny.Seq
            out17_: bool
            out18_: _dafny.Seq
            out16_, out17_, out18_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_14_closeBudget_)
            d_15_cg_ = out16_
            d_16_ci_ = out17_
            d_17_cc_ = out18_
            generated = d_15_cg_
            insideConstrainedOut = d_16_ci_
            currentConstrainedOut = d_17_cc_
            d_1_steps_ = maxSteps
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

