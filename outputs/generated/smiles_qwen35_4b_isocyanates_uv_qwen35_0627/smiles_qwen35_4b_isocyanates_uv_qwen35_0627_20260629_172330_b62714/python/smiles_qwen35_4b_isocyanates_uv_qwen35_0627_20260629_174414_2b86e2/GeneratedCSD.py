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
        d_2_spanLen_: int
        d_2_spanLen_ = 0
        d_3_minSpanLen_: int
        d_3_minSpanLen_ = 10
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate a SMILES string for a valid isocyanate molecule. Isocyanates contain the N=C=O functional group. Output a complete SMILES with a carbon-containing group bonded to N=C=O. Examples: CCCN=C=O (propyl isocyanate), CCCCN=C=O (butyl isocyanate), c1ccccc1N=C=O (phenyl isocyanate), CC(C)N=C=O (isopropyl isocyanate), C1CCCCC1N=C=O (cyclohexyl isocyanate). The R group must have at least 2 carbon atoms. Output only the SMILES string.")))
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out0_: _dafny.Seq
                        out0_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out0_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                                d_2_spanLen_ = 0
                    elif True:
                        if (d_2_spanLen_) >= (d_3_minSpanLen_):
                            d_5_closeBudget_: int
                            d_5_closeBudget_ = (maxSteps) - (d_1_steps_)
                            d_6_cg_: _dafny.Seq
                            d_7_ci_: bool
                            d_8_cc_: _dafny.Seq
                            out1_: _dafny.Seq
                            out2_: bool
                            out3_: _dafny.Seq
                            out1_, out2_, out3_ = (d_0_helpers_).CloseSpanWithinBudget(lm, parser, prompt, generated, currentConstrainedOut, eosToken, d_5_closeBudget_)
                            d_6_cg_ = out1_
                            d_7_ci_ = out2_
                            d_8_cc_ = out3_
                            generated = d_6_cg_
                            insideConstrainedOut = d_7_ci_
                            currentConstrainedOut = d_8_cc_
                            d_1_steps_ = maxSteps
                        elif True:
                            d_9_constrainedPrompt_: _dafny.Seq
                            d_9_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                            d_10_next_: _dafny.Seq = _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, ""))
                            if (d_2_spanLen_) < (5):
                                out4_: _dafny.Seq
                                out4_ = (d_0_helpers_).SafeTemperatureConstrainedStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, _dafny.BigRational('12e-1'), eosToken)
                                d_10_next_ = out4_
                            elif True:
                                out5_: _dafny.Seq
                                out5_ = (d_0_helpers_).SafeRepetitionPenaltyStep(lm, parser, d_9_constrainedPrompt_, currentConstrainedOut, generated, _dafny.BigRational('25e-1'), eosToken)
                                d_10_next_ = out5_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_10_next_) == (eosToken):
                                raise _dafny.Break("0")
                            elif True:
                                d_11_isComplete_: bool
                                d_11_isComplete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                                if d_11_isComplete_:
                                    d_12_closedGenerated_: _dafny.Seq
                                    d_13_closedInside_: bool
                                    d_14_closedCurrent_: _dafny.Seq
                                    out6_: _dafny.Seq
                                    out7_: bool
                                    out8_: _dafny.Seq
                                    out6_, out7_, out8_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_12_closedGenerated_ = out6_
                                    d_13_closedInside_ = out7_
                                    d_14_closedCurrent_ = out8_
                                    generated = d_12_closedGenerated_
                                    insideConstrainedOut = d_13_closedInside_
                                    currentConstrainedOut = d_14_closedCurrent_
                                    d_2_spanLen_ = 0
                                elif True:
                                    d_15_appendedGenerated_: _dafny.Seq
                                    d_16_appendedInside_: bool
                                    d_17_appendedCurrent_: _dafny.Seq
                                    out9_: _dafny.Seq
                                    out10_: bool
                                    out11_: _dafny.Seq
                                    out9_, out10_, out11_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_10_next_)
                                    d_15_appendedGenerated_ = out9_
                                    d_16_appendedInside_ = out10_
                                    d_17_appendedCurrent_ = out11_
                                    generated = d_15_appendedGenerated_
                                    insideConstrainedOut = d_16_appendedInside_
                                    currentConstrainedOut = d_17_appendedCurrent_
                                    d_2_spanLen_ = (d_2_spanLen_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

