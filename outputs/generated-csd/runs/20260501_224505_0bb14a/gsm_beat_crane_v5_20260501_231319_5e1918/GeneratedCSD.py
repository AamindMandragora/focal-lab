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
        d_2_switchLen_: int
        d_2_switchLen_ = 6
        d_3_flatGroups_: _dafny.Seq
        out0_: _dafny.Seq
        out0_ = VerifiedDecoderAgent.CSDHelpers.FlattenTokenGroups(validTokenGroups)
        d_3_flatGroups_ = out0_
        with _dafny.label("0"):
            while (d_1_steps_) < (maxSteps):
                with _dafny.c_label("0"):
                    if not(insideConstrainedOut):
                        d_4_next_: _dafny.Seq
                        out1_: _dafny.Seq
                        out1_ = (d_0_helpers_).UnconstrainedStep(lm, prompt, generated)
                        d_4_next_ = out1_
                        d_1_steps_ = (d_1_steps_) + (1)
                        if (d_4_next_) == (eosToken):
                            raise _dafny.Break("0")
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_next_]))
                            if (d_4_next_) == (_dafny.SeqWithoutIsStrInference(map(_dafny.CodePoint, "<<"))):
                                insideConstrainedOut = True
                                currentConstrainedOut = _dafny.SeqWithoutIsStrInference([])
                    elif True:
                        d_5_complete_: bool
                        d_5_complete_ = (parser).IsCompletePrefix(currentConstrainedOut)
                        if d_5_complete_:
                            d_6_closedGenerated_: _dafny.Seq
                            d_7_closedInside_: bool
                            d_8_closedCurrent_: _dafny.Seq
                            out2_: _dafny.Seq
                            out3_: bool
                            out4_: _dafny.Seq
                            out2_, out3_, out4_ = (d_0_helpers_).CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut)
                            d_6_closedGenerated_ = out2_
                            d_7_closedInside_ = out3_
                            d_8_closedCurrent_ = out4_
                            generated = d_6_closedGenerated_
                            insideConstrainedOut = d_7_closedInside_
                            currentConstrainedOut = d_8_closedCurrent_
                            d_1_steps_ = (d_1_steps_) + (1)
                        elif True:
                            d_9_stablePrefix_: _dafny.Seq
                            d_9_stablePrefix_ = _dafny.SeqWithoutIsStrInference((generated)[:(len(generated)) - (len(currentConstrainedOut)):])
                            if (len(currentConstrainedOut)) < (d_2_switchLen_):
                                d_10_tokensToPenalize_: _dafny.Seq
                                d_10_tokensToPenalize_ = _dafny.SeqWithoutIsStrInference([])
                                d_11_i_: int
                                d_11_i_ = 0
                                while (d_11_i_) < (len(d_3_flatGroups_)):
                                    d_12_tok_: _dafny.Seq
                                    d_12_tok_ = (d_3_flatGroups_)[d_11_i_]
                                    d_13_valid_: bool
                                    out5_: bool
                                    out5_ = (d_0_helpers_).IsTokenValidNext(parser, currentConstrainedOut, d_12_tok_)
                                    d_13_valid_ = out5_
                                    if ((d_12_tok_) in ((lm).Tokens)) and (not(d_13_valid_)):
                                        d_10_tokensToPenalize_ = (d_10_tokensToPenalize_) + (_dafny.SeqWithoutIsStrInference([d_12_tok_]))
                                    d_11_i_ = (d_11_i_) + (1)
                                d_14_next_: _dafny.Seq
                                out6_: _dafny.Seq
                                out6_ = (d_0_helpers_).PenalizedConstrainedStep(lm, parser, (prompt) + (d_9_stablePrefix_), currentConstrainedOut, d_10_tokensToPenalize_, _dafny.BigRational('5e0'), eosToken)
                                d_14_next_ = out6_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if (d_14_next_) == (eosToken):
                                    raise _dafny.Break("0")
                                elif True:
                                    d_15_appendedGenerated_: _dafny.Seq
                                    d_16_appendedInside_: bool
                                    d_17_appendedCurrent_: _dafny.Seq
                                    out7_: _dafny.Seq
                                    out8_: bool
                                    out9_: _dafny.Seq
                                    out7_, out8_, out9_ = (d_0_helpers_).AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, d_14_next_)
                                    d_15_appendedGenerated_ = out7_
                                    d_16_appendedInside_ = out8_
                                    d_17_appendedCurrent_ = out9_
                                    generated = d_15_appendedGenerated_
                                    insideConstrainedOut = d_16_appendedInside_
                                    currentConstrainedOut = d_17_appendedCurrent_
                            elif True:
                                d_18_steppedGenerated_: _dafny.Seq
                                d_19_steppedInside_: bool
                                d_20_steppedCurrent_: _dafny.Seq
                                d_21_hitEos_: bool
                                out10_: _dafny.Seq
                                out11_: bool
                                out12_: _dafny.Seq
                                out13_: bool
                                out10_, out11_, out12_, out13_ = (d_0_helpers_).ConstrainedStep(lm, parser, (prompt) + (d_9_stablePrefix_), generated, currentConstrainedOut, eosToken)
                                d_18_steppedGenerated_ = out10_
                                d_19_steppedInside_ = out11_
                                d_20_steppedCurrent_ = out12_
                                d_21_hitEos_ = out13_
                                d_1_steps_ = (d_1_steps_) + (1)
                                if d_21_hitEos_:
                                    raise _dafny.Break("0")
                                elif True:
                                    generated = d_18_steppedGenerated_
                                    insideConstrainedOut = d_19_steppedInside_
                                    currentConstrainedOut = d_20_steppedCurrent_
                    pass
            pass
        cost = d_1_steps_
        return generated, insideConstrainedOut, currentConstrainedOut, cost

