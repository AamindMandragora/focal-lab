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
        (d_0_helpers_).AppendTaskGuidance(lm, _dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "Symbolic math problem. Variables appear in {curly braces} like {n}, {x}, {k}. RULES: (1) Inside every <<...>> write ONE short arithmetic expression with BARE variable names — strip the braces: write n, x, k (NOT {n}, {x}, {k}). (2) Use only +, -, *, /, // (integer division), int(), and parentheses; NO English words, NO nested <<, NO equals signs. (3) Close every << with a matching >>. (4) Be concise. End with exactly: The final answer is <<EXPR>>. Example problem: 'A box has {x} items per shelf and {n} shelves; total?' Solution: '<<n*x>>. The final answer is <<n*x>>.'")))
        d_1_steps_: int
        d_1_steps_ = 0
        d_2_spanLen_: int
        d_2_spanLen_ = 0
        d_3_spanCap_: int
        d_3_spanCap_ = 40
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
                            d_5_triggerOpen_: bool
                            d_5_triggerOpen_ = (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<")))
                            if (((not(d_5_triggerOpen_)) and ((d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<"))))) and ((len(generated)) >= (2))) and (((generated)[(len(generated)) - (2)]) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<")))):
                                d_5_triggerOpen_ = True
                            if d_5_triggerOpen_:
                                d_6_enteredGenerated_: _dafny.Seq
                                d_7_enteredInside_: bool
                                d_8_enteredCurrent_: _dafny.Seq
                                out1_: _dafny.Seq
                                out2_: bool
                                out3_: _dafny.Seq
                                out1_, out2_, out3_ = (d_0_helpers_).EnterObservedConstrainedSpan(lm, generated)
                                d_6_enteredGenerated_ = out1_
                                d_7_enteredInside_ = out2_
                                d_8_enteredCurrent_ = out3_
                                generated = d_6_enteredGenerated_
                                insideConstrainedOut = d_7_enteredInside_
                                currentConstrainedOut = d_8_enteredCurrent_
                                d_2_spanLen_ = 0
                    elif (parser).IsCompletePrefix(currentConstrainedOut):
                        d_9_closedGenerated_: _dafny.Seq
                        d_10_closedInside_: bool
                        d_11_closedCurrent_: _dafny.Seq
                        out4_: _dafny.Seq
                        out5_: bool
                        out6_: _dafny.Seq
                        out4_, out5_, out6_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                        d_9_closedGenerated_ = out4_
                        d_10_closedInside_ = out5_
                        d_11_closedCurrent_ = out6_
                        generated = d_9_closedGenerated_
                        insideConstrainedOut = d_10_closedInside_
                        currentConstrainedOut = d_11_closedCurrent_
                        d_1_steps_ = (d_1_steps_) + (1)
                        d_2_spanLen_ = 0
                    elif (d_2_spanLen_) >= (d_3_spanCap_):
                        raise _dafny.Break("0")
                    elif True:
                        d_12_constrainedPrompt_: _dafny.Seq
                        d_12_constrainedPrompt_ = (prompt) + (_dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):]))
                        d_13_next_: _dafny.Seq
                        out7_: _dafny.Seq
                        out7_ = (d_0_helpers_).AdaptiveConstrainedStep(lm, parser, d_12_constrainedPrompt_, currentConstrainedOut, validTokenGroups, _dafny.BigRational('4e0'), 12, eosToken)
                        d_13_next_ = out7_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_13_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            d_14_appendedGenerated_: _dafny.Seq
                            d_15_appendedInside_: bool
                            d_16_appendedCurrent_: _dafny.Seq
                            out8_: _dafny.Seq
                            out9_: bool
                            out10_: _dafny.Seq
                            out8_, out9_, out10_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_13_next_)
                            d_14_appendedGenerated_ = out8_
                            d_15_appendedInside_ = out9_
                            d_16_appendedCurrent_ = out10_
                            generated = d_14_appendedGenerated_
                            insideConstrainedOut = d_15_appendedInside_
                            currentConstrainedOut = d_16_appendedCurrent_
                            d_2_spanLen_ = (d_2_spanLen_) + (1)
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

