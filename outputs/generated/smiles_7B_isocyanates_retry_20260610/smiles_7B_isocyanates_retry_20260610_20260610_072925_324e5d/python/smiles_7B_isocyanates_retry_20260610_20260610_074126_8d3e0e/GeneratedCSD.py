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
        if (maxSteps) == (0):
            pass
        elif True:
            (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Generate the canonical SMILES string for the chemical compound or class in the prompt. Output ONLY the SMILES notation. Example: for 'ethanol' output 'CCO', for 'benzene' output 'c1ccccc1', for 'isocyanates' output a molecule with the -N=C=O group.")))
            d_1_steps_: int
            d_1_steps_ = 0
            if not(insideConstrainedOut):
                if (d_1_steps_) < (maxSteps):
                    d_2_openGenerated_: _dafny.Seq
                    d_3_openInside_: bool
                    d_4_openCurrent_: _dafny.Seq
                    out0_: _dafny.Seq
                    out1_: bool
                    out2_: _dafny.Seq
                    out0_, out1_, out2_ = (d_0_helpers_).OpenConstrainedSpan(lm, generated)
                    d_2_openGenerated_ = out0_
                    d_3_openInside_ = out1_
                    d_4_openCurrent_ = out2_
                    generated = d_2_openGenerated_
                    insideConstrainedOut = d_3_openInside_
                    currentConstrainedOut = d_4_openCurrent_
                    d_1_steps_ = (d_1_steps_) + (1)
            with _dafny.label("1_0"):
                while ((d_1_steps_) < (maxSteps)) and (insideConstrainedOut):
                    with _dafny.c_label("1_0"):
                        d_5_cg_: _dafny.Seq
                        d_6_ci_: bool
                        d_7_cc_: _dafny.Seq
                        d_8_closed_: bool
                        out3_: _dafny.Seq
                        out4_: bool
                        out5_: _dafny.Seq
                        out6_: bool
                        out3_, out4_, out5_, out6_ = (d_0_helpers_).CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut)
                        d_5_cg_ = out3_
                        d_6_ci_ = out4_
                        d_7_cc_ = out5_
                        d_8_closed_ = out6_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if d_8_closed_:
                            generated = d_5_cg_
                            insideConstrainedOut = d_6_ci_
                            currentConstrainedOut = d_7_cc_
                        elif (d_1_steps_) < (maxSteps):
                            d_9_next_: _dafny.Seq
                            out7_: _dafny.Seq
                            out7_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken)
                            d_9_next_ = out7_
                            d_1_steps_ = (d_1_steps_) + (1)
                            if (d_9_next_) == (eosToken):
                                if ((parser).IsCompletePrefix(currentConstrainedOut)) and ((d_1_steps_) < (maxSteps)):
                                    d_10_closedGenerated_: _dafny.Seq
                                    d_11_closedInside_: bool
                                    d_12_closedCurrent_: _dafny.Seq
                                    out8_: _dafny.Seq
                                    out9_: bool
                                    out10_: _dafny.Seq
                                    out8_, out9_, out10_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                                    d_10_closedGenerated_ = out8_
                                    d_11_closedInside_ = out9_
                                    d_12_closedCurrent_ = out10_
                                    generated = d_10_closedGenerated_
                                    insideConstrainedOut = d_11_closedInside_
                                    currentConstrainedOut = d_12_closedCurrent_
                                    d_1_steps_ = (d_1_steps_) + (1)
                                raise _dafny.Break("1_0")
                            elif True:
                                d_13_appendedGenerated_: _dafny.Seq
                                d_14_appendedInside_: bool
                                d_15_appendedCurrent_: _dafny.Seq
                                out11_: _dafny.Seq
                                out12_: bool
                                out13_: _dafny.Seq
                                out11_, out12_, out13_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_9_next_)
                                d_13_appendedGenerated_ = out11_
                                d_14_appendedInside_ = out12_
                                d_15_appendedCurrent_ = out13_
                                generated = d_13_appendedGenerated_
                                insideConstrainedOut = d_14_appendedInside_
                                currentConstrainedOut = d_15_appendedCurrent_
                        pass
                pass
            if ((insideConstrainedOut) and ((parser).IsCompletePrefix(currentConstrainedOut))) and ((d_1_steps_) < (maxSteps)):
                d_16_closedGenerated_: _dafny.Seq
                d_17_closedInside_: bool
                d_18_closedCurrent_: _dafny.Seq
                out14_: _dafny.Seq
                out15_: bool
                out16_: _dafny.Seq
                out14_, out15_, out16_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                d_16_closedGenerated_ = out14_
                d_17_closedInside_ = out15_
                d_18_closedCurrent_ = out16_
                generated = d_16_closedGenerated_
                insideConstrainedOut = d_17_closedInside_
                currentConstrainedOut = d_18_closedCurrent_
                d_1_steps_ = (d_1_steps_) + (1)
            cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

