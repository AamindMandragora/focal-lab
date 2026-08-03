import pytest

from synthesis.evaluate.run_legacy_fixed_strategy import (
    _install_itergen_transformers_compat,
)


class _GenerationConfig:
    def __init__(self, *, do_sample: bool):
        self.do_sample = do_sample

    def update(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


class _ModernModel:
    pass


class _OldModel:
    def __init__(self):
        self.calls = []

    def _get_logits_warper(self, generation_config, *, device):
        self.calls.append((generation_config, device))
        return "legacy-warper"


class _LegacyIterGen:
    def update_gen_args(self, **gen_args):
        self.generation_config.update(**gen_args)
        self.logit_warper = self.model._get_logits_warper(
            self.generation_config,
            device=self.device,
        )


def _instance(*, do_sample: bool):
    value = _LegacyIterGen()
    value.generation_config = _GenerationConfig(do_sample=do_sample)
    value.model = _ModernModel()
    value.device = "cuda:0"
    return value


def test_modern_transformers_greedy_itergen_uses_identity_warper():
    _install_itergen_transformers_compat(_LegacyIterGen)
    value = _instance(do_sample=False)

    value.update_gen_args(max_new_tokens=128)

    assert value.generation_config.max_new_tokens == 128
    assert list(value.logit_warper) == []


def test_modern_transformers_sampling_fails_instead_of_changing_baseline_semantics():
    _install_itergen_transformers_compat(_LegacyIterGen)
    value = _instance(do_sample=True)

    with pytest.raises(RuntimeError, match="do_sample=False"):
        value.update_gen_args(max_new_tokens=128)


def test_older_transformers_keeps_the_original_logits_warper_path():
    class OldIterGen:
        def update_gen_args(self, **gen_args):
            raise AssertionError("the compatibility wrapper should replace this method")

    _install_itergen_transformers_compat(OldIterGen)
    value = OldIterGen()
    value.generation_config = _GenerationConfig(do_sample=False)
    value.model = _OldModel()
    value.device = "cuda:2"

    value.update_gen_args(max_new_tokens=64)

    assert value.logit_warper == "legacy-warper"
    assert value.model.calls == [(value.generation_config, "cuda:2")]
