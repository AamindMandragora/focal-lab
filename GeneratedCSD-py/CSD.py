import sys
from typing import Callable, Any, TypeVar, NamedTuple
from math import floor
from itertools import count

import module_ as module_
import _dafny as _dafny
import System_ as System_
import Decoding as Decoding

# Module: CSD

class default__:
    def  __init__(self):
        pass

    @staticmethod
    def AllowedNext(policy, st, prefix):
        source0_ = policy
        if True:
            if source0_.is_Base:
                return Decoding.default__.Vocabulary()
        if True:
            if source0_.is_MaskWithSynCode:
                d_0_base_ = source0_.base
                return (default__.AllowedNext(d_0_base_, st, prefix)).intersection(Decoding.default__.Parser__AllowedNext(prefix))
        if True:
            if source0_.is_MaskWithCRANE:
                d_1_base_ = source0_.base
                return (default__.AllowedNext(d_1_base_, st, prefix)).intersection(Decoding.default__.Parser__AllowedNext(prefix))
        if True:
            if source0_.is_WithRejection:
                d_2_base_ = source0_.base
                def iife0_():
                    coll0_ = _dafny.Set()
                    compr_0_: _dafny.Seq
                    for compr_0_ in (default__.AllowedNext(d_2_base_, st, prefix)).Elements:
                        d_3_t_: _dafny.Seq = compr_0_
                        if ((d_3_t_) in (default__.AllowedNext(d_2_base_, st, prefix))) and (Decoding.default__.Accept(prefix, d_3_t_)):
                            coll0_ = coll0_.union(_dafny.Set([d_3_t_]))
                    return _dafny.Set(coll0_)
                return iife0_()
                
        if True:
            if source0_.is_Fallback:
                d_4_primary_ = source0_.primary
                d_5_backup_ = source0_.backup
                if (len(default__.AllowedNext(d_4_primary_, st, prefix))) > (0):
                    return default__.AllowedNext(d_4_primary_, st, prefix)
                elif True:
                    return default__.AllowedNext(d_5_backup_, st, prefix)
        if True:
            if source0_.is_Intersect:
                d_6_a_ = source0_.a
                d_7_b_ = source0_.b
                return (default__.AllowedNext(d_6_a_, st, prefix)).intersection(default__.AllowedNext(d_7_b_, st, prefix))
        if True:
            d_8_a_ = source0_.a
            d_9_b_ = source0_.b
            return (default__.AllowedNext(d_8_a_, st, prefix)) | (default__.AllowedNext(d_9_b_, st, prefix))

    @staticmethod
    def IsGrammarAligned(policy):
        source0_ = policy
        if True:
            if source0_.is_MaskWithSynCode:
                return True
        if True:
            if source0_.is_MaskWithCRANE:
                return True
        if True:
            if source0_.is_WithRejection:
                d_0_b_ = source0_.base
                return default__.IsGrammarAligned(d_0_b_)
        if True:
            if source0_.is_Fallback:
                d_1_p_ = source0_.primary
                d_2_b_ = source0_.backup
                return (default__.IsGrammarAligned(d_1_p_)) and (default__.IsGrammarAligned(d_2_b_))
        if True:
            if source0_.is_Intersect:
                d_3_a_ = source0_.a
                d_4_b_ = source0_.b
                return (default__.IsGrammarAligned(d_3_a_)) and (default__.IsGrammarAligned(d_4_b_))
        if True:
            if source0_.is_Union:
                return False
        if True:
            return False

    @staticmethod
    def IsSemFiltered(policy):
        source0_ = policy
        if True:
            if source0_.is_WithRejection:
                return True
        if True:
            if source0_.is_MaskWithSynCode:
                d_0_b_ = source0_.base
                return default__.IsSemFiltered(d_0_b_)
        if True:
            if source0_.is_MaskWithCRANE:
                d_1_b_ = source0_.base
                return default__.IsSemFiltered(d_1_b_)
        if True:
            if source0_.is_Fallback:
                d_2_p_ = source0_.primary
                d_3_b_ = source0_.backup
                return (default__.IsSemFiltered(d_2_p_)) and (default__.IsSemFiltered(d_3_b_))
        if True:
            if source0_.is_Intersect:
                d_4_a_ = source0_.a
                d_5_b_ = source0_.b
                return (default__.IsSemFiltered(d_4_a_)) or (default__.IsSemFiltered(d_5_b_))
        if True:
            if source0_.is_Union:
                d_6_a_ = source0_.a
                d_7_b_ = source0_.b
                return (default__.IsSemFiltered(d_6_a_)) or (default__.IsSemFiltered(d_7_b_))
        if True:
            return False

    @staticmethod
    def RunAttempt(g, attempt, prompt, maxSteps):
        source0_ = attempt
        if True:
            if source0_.is_Unconstrained:
                d_0_strategy_ = source0_.strategy
                return default__.LLM__Generate(prompt, maxSteps, d_0_strategy_)
        if True:
            if source0_.is_Constrained:
                d_1_policy_ = source0_.policy
                return default__.ConstrainedGenerate(g, d_1_policy_, prompt, maxSteps)
        if True:
            if source0_.is_Repair:
                d_2_base_ = source0_.base
                return default__.RepairTransform(default__.RunAttempt(g, d_2_base_, prompt, maxSteps))
        if True:
            d_3_policy_ = source0_.policy
            d_4_beamWidth_ = source0_.beamWidth
            return default__.ConstrainedSearchGenerate(g, d_3_policy_, d_4_beamWidth_, prompt, maxSteps)

    @staticmethod
    def Passes(check, g, s):
        source0_ = check
        if True:
            if source0_.is_ParseOnly:
                return default__.ParseOk(g, s)
        if True:
            return (default__.ParseOk(g, s)) and (default__.SemanticOk(g, s))

    @staticmethod
    def Size(program):
        d_0___accumulator_ = 0
        while True:
            with _dafny.label():
                source0_ = program
                if True:
                    if source0_.is_ReturnParsed:
                        return 1
                if True:
                    if source0_.is_TryThenElse:
                        d_1_onFail_ = source0_.onFail
                        return (1) + (default__.Size(d_1_onFail_))
                if True:
                    if source0_.is_TryK:
                        d_2_onFail_ = source0_.onFail
                        return (1) + (default__.Size(d_2_onFail_))
                if True:
                    if source0_.is_BestOfNThenElse:
                        d_3_onFail_ = source0_.onFail
                        return (1) + (default__.Size(d_3_onFail_))
                if True:
                    d_4_onFail_ = source0_.onFail
                    return (1) + (default__.Size(d_4_onFail_))
                break

    @staticmethod
    def Run(program, prompt, maxSteps):
        while True:
            with _dafny.label():
                source0_ = program
                if True:
                    if source0_.is_ReturnParsed:
                        d_0_g_ = source0_.g
                        d_1_policy_ = source0_.policy
                        return default__.ConstrainedGenerate(d_0_g_, d_1_policy_, prompt, maxSteps)
                if True:
                    if source0_.is_CompleteIfPrefixOkElse:
                        d_2_g_ = source0_.g
                        d_3_strategy_ = source0_.strategy
                        d_4_policy_ = source0_.policy
                        d_5_onFail_ = source0_.onFail
                        d_6_s_ = default__.LLM__Generate(prompt, maxSteps, d_3_strategy_)
                        if default__.ParseOkPrefix(d_2_g_, d_6_s_):
                            return default__.ConstrainedCompleteFromPrefix(d_2_g_, d_4_policy_, d_6_s_, maxSteps)
                        elif True:
                            in0_ = d_5_onFail_
                            in1_ = prompt
                            in2_ = maxSteps
                            program = in0_
                            prompt = in1_
                            maxSteps = in2_
                            raise _dafny.TailCall()
                if True:
                    if source0_.is_TryThenElse:
                        d_7_g_ = source0_.g
                        d_8_attempt_ = source0_.attempt
                        d_9_check_ = source0_.check
                        d_10_onFail_ = source0_.onFail
                        d_11_s_ = default__.RunAttempt(d_7_g_, d_8_attempt_, prompt, maxSteps)
                        if default__.Passes(d_9_check_, d_7_g_, d_11_s_):
                            return d_11_s_
                        elif True:
                            in3_ = d_10_onFail_
                            in4_ = prompt
                            in5_ = maxSteps
                            program = in3_
                            prompt = in4_
                            maxSteps = in5_
                            raise _dafny.TailCall()
                if True:
                    if source0_.is_TryK:
                        d_12_g_ = source0_.g
                        d_13_k_ = source0_.k
                        d_14_attempt_ = source0_.attempt
                        d_15_check_ = source0_.check
                        d_16_onFail_ = source0_.onFail
                        if (d_13_k_) == (0):
                            in6_ = d_16_onFail_
                            in7_ = prompt
                            in8_ = maxSteps
                            program = in6_
                            prompt = in7_
                            maxSteps = in8_
                            raise _dafny.TailCall()
                        elif True:
                            d_17_s_ = default__.RunAttempt(d_12_g_, d_14_attempt_, prompt, maxSteps)
                            if default__.Passes(d_15_check_, d_12_g_, d_17_s_):
                                return d_17_s_
                            elif True:
                                in9_ = Program_TryK(d_12_g_, (d_13_k_) - (1), d_14_attempt_, d_15_check_, d_16_onFail_)
                                in10_ = prompt
                                in11_ = maxSteps
                                program = in9_
                                prompt = in10_
                                maxSteps = in11_
                                raise _dafny.TailCall()
                if True:
                    d_18_g_ = source0_.g
                    d_19_n_ = source0_.n
                    d_20_strategy_ = source0_.strategy
                    d_21_check_ = source0_.check
                    d_22_onFail_ = source0_.onFail
                    d_23_r_ = default__.BestOfNSelectPassing(d_18_g_, d_19_n_, d_20_strategy_, d_21_check_, prompt, maxSteps)
                    if (d_23_r_).found:
                        return (d_23_r_).s
                    elif True:
                        in12_ = d_22_onFail_
                        in13_ = prompt
                        in14_ = maxSteps
                        program = in12_
                        prompt = in13_
                        maxSteps = in14_
                        raise _dafny.TailCall()
                break


class Policy:
    @classmethod
    def default(cls, ):
        return lambda: Policy_Base(Decoding.ProposalStrategy.default()())
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_Base(self) -> bool:
        return isinstance(self, Policy_Base)
    @property
    def is_MaskWithSynCode(self) -> bool:
        return isinstance(self, Policy_MaskWithSynCode)
    @property
    def is_MaskWithCRANE(self) -> bool:
        return isinstance(self, Policy_MaskWithCRANE)
    @property
    def is_WithRejection(self) -> bool:
        return isinstance(self, Policy_WithRejection)
    @property
    def is_Fallback(self) -> bool:
        return isinstance(self, Policy_Fallback)
    @property
    def is_Intersect(self) -> bool:
        return isinstance(self, Policy_Intersect)
    @property
    def is_Union(self) -> bool:
        return isinstance(self, Policy_Union)

class Policy_Base(Policy, NamedTuple('Base', [('strategy', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Policy.Base({_dafny.string_of(self.strategy)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Policy_Base) and self.strategy == __o.strategy
    def __hash__(self) -> int:
        return super().__hash__()

class Policy_MaskWithSynCode(Policy, NamedTuple('MaskWithSynCode', [('base', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Policy.MaskWithSynCode({_dafny.string_of(self.base)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Policy_MaskWithSynCode) and self.base == __o.base
    def __hash__(self) -> int:
        return super().__hash__()

class Policy_MaskWithCRANE(Policy, NamedTuple('MaskWithCRANE', [('base', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Policy.MaskWithCRANE({_dafny.string_of(self.base)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Policy_MaskWithCRANE) and self.base == __o.base
    def __hash__(self) -> int:
        return super().__hash__()

class Policy_WithRejection(Policy, NamedTuple('WithRejection', [('base', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Policy.WithRejection({_dafny.string_of(self.base)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Policy_WithRejection) and self.base == __o.base
    def __hash__(self) -> int:
        return super().__hash__()

class Policy_Fallback(Policy, NamedTuple('Fallback', [('primary', Any), ('backup', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Policy.Fallback({_dafny.string_of(self.primary)}, {_dafny.string_of(self.backup)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Policy_Fallback) and self.primary == __o.primary and self.backup == __o.backup
    def __hash__(self) -> int:
        return super().__hash__()

class Policy_Intersect(Policy, NamedTuple('Intersect', [('a', Any), ('b', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Policy.Intersect({_dafny.string_of(self.a)}, {_dafny.string_of(self.b)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Policy_Intersect) and self.a == __o.a and self.b == __o.b
    def __hash__(self) -> int:
        return super().__hash__()

class Policy_Union(Policy, NamedTuple('Union', [('a', Any), ('b', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Policy.Union({_dafny.string_of(self.a)}, {_dafny.string_of(self.b)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Policy_Union) and self.a == __o.a and self.b == __o.b
    def __hash__(self) -> int:
        return super().__hash__()


class State:
    @_dafny.classproperty
    def AllSingletonConstructors(cls):
        return [State_State()]
    @classmethod
    def default(cls, ):
        return lambda: State_State()
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_State(self) -> bool:
        return isinstance(self, State_State)

class State_State(State, NamedTuple('State', [])):
    def __dafnystr__(self) -> str:
        return f'CSD.State.State'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, State_State)
    def __hash__(self) -> int:
        return super().__hash__()


class Attempt:
    @classmethod
    def default(cls, ):
        return lambda: Attempt_Unconstrained(Decoding.ProposalStrategy.default()())
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_Unconstrained(self) -> bool:
        return isinstance(self, Attempt_Unconstrained)
    @property
    def is_Constrained(self) -> bool:
        return isinstance(self, Attempt_Constrained)
    @property
    def is_Repair(self) -> bool:
        return isinstance(self, Attempt_Repair)
    @property
    def is_ConstrainedSearch(self) -> bool:
        return isinstance(self, Attempt_ConstrainedSearch)

class Attempt_Unconstrained(Attempt, NamedTuple('Unconstrained', [('strategy', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Attempt.Unconstrained({_dafny.string_of(self.strategy)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Attempt_Unconstrained) and self.strategy == __o.strategy
    def __hash__(self) -> int:
        return super().__hash__()

class Attempt_Constrained(Attempt, NamedTuple('Constrained', [('policy', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Attempt.Constrained({_dafny.string_of(self.policy)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Attempt_Constrained) and self.policy == __o.policy
    def __hash__(self) -> int:
        return super().__hash__()

class Attempt_Repair(Attempt, NamedTuple('Repair', [('base', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Attempt.Repair({_dafny.string_of(self.base)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Attempt_Repair) and self.base == __o.base
    def __hash__(self) -> int:
        return super().__hash__()

class Attempt_ConstrainedSearch(Attempt, NamedTuple('ConstrainedSearch', [('policy', Any), ('beamWidth', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Attempt.ConstrainedSearch({_dafny.string_of(self.policy)}, {_dafny.string_of(self.beamWidth)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Attempt_ConstrainedSearch) and self.policy == __o.policy and self.beamWidth == __o.beamWidth
    def __hash__(self) -> int:
        return super().__hash__()


class Check:
    @_dafny.classproperty
    def AllSingletonConstructors(cls):
        return [Check_ParseOnly(), Check_ParseAndSemantic()]
    @classmethod
    def default(cls, ):
        return lambda: Check_ParseOnly()
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_ParseOnly(self) -> bool:
        return isinstance(self, Check_ParseOnly)
    @property
    def is_ParseAndSemantic(self) -> bool:
        return isinstance(self, Check_ParseAndSemantic)

class Check_ParseOnly(Check, NamedTuple('ParseOnly', [])):
    def __dafnystr__(self) -> str:
        return f'CSD.Check.ParseOnly'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Check_ParseOnly)
    def __hash__(self) -> int:
        return super().__hash__()

class Check_ParseAndSemantic(Check, NamedTuple('ParseAndSemantic', [])):
    def __dafnystr__(self) -> str:
        return f'CSD.Check.ParseAndSemantic'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Check_ParseAndSemantic)
    def __hash__(self) -> int:
        return super().__hash__()


class Program:
    @classmethod
    def default(cls, ):
        return lambda: Program_ReturnParsed(int(0), Policy.default()())
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_TryThenElse(self) -> bool:
        return isinstance(self, Program_TryThenElse)
    @property
    def is_TryK(self) -> bool:
        return isinstance(self, Program_TryK)
    @property
    def is_BestOfNThenElse(self) -> bool:
        return isinstance(self, Program_BestOfNThenElse)
    @property
    def is_CompleteIfPrefixOkElse(self) -> bool:
        return isinstance(self, Program_CompleteIfPrefixOkElse)
    @property
    def is_ReturnParsed(self) -> bool:
        return isinstance(self, Program_ReturnParsed)

class Program_TryThenElse(Program, NamedTuple('TryThenElse', [('g', Any), ('attempt', Any), ('check', Any), ('onFail', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Program.TryThenElse({_dafny.string_of(self.g)}, {_dafny.string_of(self.attempt)}, {_dafny.string_of(self.check)}, {_dafny.string_of(self.onFail)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Program_TryThenElse) and self.g == __o.g and self.attempt == __o.attempt and self.check == __o.check and self.onFail == __o.onFail
    def __hash__(self) -> int:
        return super().__hash__()

class Program_TryK(Program, NamedTuple('TryK', [('g', Any), ('k', Any), ('attempt', Any), ('check', Any), ('onFail', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Program.TryK({_dafny.string_of(self.g)}, {_dafny.string_of(self.k)}, {_dafny.string_of(self.attempt)}, {_dafny.string_of(self.check)}, {_dafny.string_of(self.onFail)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Program_TryK) and self.g == __o.g and self.k == __o.k and self.attempt == __o.attempt and self.check == __o.check and self.onFail == __o.onFail
    def __hash__(self) -> int:
        return super().__hash__()

class Program_BestOfNThenElse(Program, NamedTuple('BestOfNThenElse', [('g', Any), ('n', Any), ('strategy', Any), ('check', Any), ('onFail', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Program.BestOfNThenElse({_dafny.string_of(self.g)}, {_dafny.string_of(self.n)}, {_dafny.string_of(self.strategy)}, {_dafny.string_of(self.check)}, {_dafny.string_of(self.onFail)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Program_BestOfNThenElse) and self.g == __o.g and self.n == __o.n and self.strategy == __o.strategy and self.check == __o.check and self.onFail == __o.onFail
    def __hash__(self) -> int:
        return super().__hash__()

class Program_CompleteIfPrefixOkElse(Program, NamedTuple('CompleteIfPrefixOkElse', [('g', Any), ('strategy', Any), ('policy', Any), ('onFail', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Program.CompleteIfPrefixOkElse({_dafny.string_of(self.g)}, {_dafny.string_of(self.strategy)}, {_dafny.string_of(self.policy)}, {_dafny.string_of(self.onFail)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Program_CompleteIfPrefixOkElse) and self.g == __o.g and self.strategy == __o.strategy and self.policy == __o.policy and self.onFail == __o.onFail
    def __hash__(self) -> int:
        return super().__hash__()

class Program_ReturnParsed(Program, NamedTuple('ReturnParsed', [('g', Any), ('policy', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.Program.ReturnParsed({_dafny.string_of(self.g)}, {_dafny.string_of(self.policy)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Program_ReturnParsed) and self.g == __o.g and self.policy == __o.policy
    def __hash__(self) -> int:
        return super().__hash__()


class SelectResult:
    @classmethod
    def default(cls, ):
        return lambda: SelectResult_Select(False, _dafny.Seq({}))
    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)
    @property
    def is_Select(self) -> bool:
        return isinstance(self, SelectResult_Select)

class SelectResult_Select(SelectResult, NamedTuple('Select', [('found', Any), ('s', Any)])):
    def __dafnystr__(self) -> str:
        return f'CSD.SelectResult.Select({_dafny.string_of(self.found)}, {_dafny.string_of(self.s)})'
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, SelectResult_Select) and self.found == __o.found and self.s == __o.s
    def __hash__(self) -> int:
        return super().__hash__()

