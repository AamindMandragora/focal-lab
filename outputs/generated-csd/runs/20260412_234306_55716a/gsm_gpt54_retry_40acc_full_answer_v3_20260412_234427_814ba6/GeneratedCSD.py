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
    def MyCSDStrategy(lm, parser, prompt, currentPrefix, maxSteps, eosToken):
        generated: _dafny.Seq = _dafny.Seq({})
        cost: int = int(0)
        d_0_helpers_: VerifiedDecoderAgent.CSDHelpers
        nw0_ = VerifiedDecoderAgent.CSDHelpers()
        nw0_.ctor__()
        d_0_helpers_ = nw0_
        generated = currentPrefix
        (d_0_helpers_).cost = 0
        cost = 0
        d_1_suffix_: _dafny.Seq
        d_1_suffix_ = _dafny.SeqWithoutIsStrInference([])
        d_2_steps_: int
        d_2_steps_ = 0
        if ((maxSteps) == (0)) or ((parser).IsCompletePrefix(generated)):
            cost = d_0_helpers_.cost
        elif True:
            d_3_emitted_: bool
            d_3_emitted_ = False
            if (not((parser).IsCompletePrefix(generated))) and ((d_2_steps_) < (maxSteps)):
                d_4_first_: _dafny.Seq
                out0_: _dafny.Seq
                out0_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, eosToken)
                d_4_first_ = out0_
                if (d_4_first_) == (eosToken):
                    d_5_firstCount_: int
                    out1_: int
                    out1_ = (d_0_helpers_).ValidTokenCount(parser, generated)
                    d_5_firstCount_ = out1_
                    if (d_5_firstCount_) > (0):
                        d_6_forced_: _dafny.Seq
                        out2_: _dafny.Seq
                        out2_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, eosToken)
                        d_6_forced_ = out2_
                        if (d_6_forced_) == (eosToken):
                            pass
                        elif True:
                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_6_forced_]))
                            d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_6_forced_]))
                            d_2_steps_ = (d_2_steps_) + (1)
                            d_3_emitted_ = True
                elif True:
                    generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_4_first_]))
                    d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_4_first_]))
                    d_2_steps_ = (d_2_steps_) + (1)
                    d_3_emitted_ = True
            with _dafny.label("1_0"):
                while (d_2_steps_) < (maxSteps):
                    with _dafny.c_label("1_0"):
                        if (parser).IsCompletePrefix(generated):
                            raise _dafny.Break("1_0")
                        elif True:
                            d_7_next_: _dafny.Seq
                            out3_: _dafny.Seq
                            out3_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, eosToken)
                            d_7_next_ = out3_
                            if (d_7_next_) == (eosToken):
                                if d_3_emitted_:
                                    raise _dafny.Break("1_0")
                                elif True:
                                    d_8_count_: int
                                    out4_: int
                                    out4_ = (d_0_helpers_).ValidTokenCount(parser, generated)
                                    d_8_count_ = out4_
                                    if (d_8_count_) > (0):
                                        d_9_forcedNext_: _dafny.Seq
                                        out5_: _dafny.Seq
                                        out5_ = (d_0_helpers_).ConstrainedStep(lm, parser, prompt, generated, eosToken)
                                        d_9_forcedNext_ = out5_
                                        if (d_9_forcedNext_) == (eosToken):
                                            raise _dafny.Break("1_0")
                                        elif True:
                                            generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_9_forcedNext_]))
                                            d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_9_forcedNext_]))
                                            d_2_steps_ = (d_2_steps_) + (1)
                                            d_3_emitted_ = True
                                    elif True:
                                        raise _dafny.Break("1_0")
                            elif True:
                                generated = (generated) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                d_1_suffix_ = (d_1_suffix_) + (_dafny.SeqWithoutIsStrInference([d_7_next_]))
                                d_2_steps_ = (d_2_steps_) + (1)
                                d_3_emitted_ = True
                        pass
                pass
            cost = d_2_steps_
        return generated, cost

