import pytest

from synthesis.evaluate.run_legacy_fixed_strategy import (
    _crane_adaptive_surface,
    _itergen_generation_kwargs,
    _itergen_generate_spider,
    _install_itergen_transformers_compat,
    _legacy_fixed_max_new_tokens,
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


def test_modern_transformers_restores_legacy_greedy_beam_defaults():
    _install_itergen_transformers_compat(_LegacyIterGen)
    value = _instance(do_sample=False)
    value.generation_config.num_beams = None
    value.generation_config.num_beam_groups = None

    value.update_gen_args(max_new_tokens=128)

    assert value.generation_config.num_beams == 1
    assert value.generation_config.num_beam_groups == 1


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


def test_qwen35_itergen_rebuilds_its_cache_with_linear_attention_layers():
    class TextConfig:
        num_hidden_layers = 2
        layer_types = ["linear_attention", "full_attention"]
        sliding_window = None
        attention_chunk_size = None

    class ModelConfig:
        def get_text_config(self, *, decoder):
            assert decoder is True
            return TextConfig()

    class HybridModel:
        config = ModelConfig()

    class HybridIterGen:
        def update_gen_args(self, **gen_args):
            raise AssertionError("not used by this test")

        def start(self, prompt):
            self.model_kwargs = {"past_key_values": "legacy-lazy-cache"}
            return prompt

    _install_itergen_transformers_compat(HybridIterGen)
    value = HybridIterGen()
    value.model = HybridModel()

    assert value.start("prompt") == "prompt"
    assert value.model_kwargs["past_key_values"].has_previous_state() is False


def test_full_attention_itergen_keeps_its_existing_cache_path():
    class TextConfig:
        layer_types = ["full_attention"]

    class ModelConfig:
        def get_text_config(self, *, decoder):
            return TextConfig()

    class FullAttentionModel:
        config = ModelConfig()

    class FullAttentionIterGen:
        def update_gen_args(self, **gen_args):
            raise AssertionError("not used by this test")

        def start(self, prompt):
            self.model_kwargs = {"past_key_values": "legacy-lazy-cache"}

    _install_itergen_transformers_compat(FullAttentionIterGen)
    value = FullAttentionIterGen()
    value.model = FullAttentionModel()

    value.start("prompt")

    assert value.model_kwargs["past_key_values"] == "legacy-lazy-cache"


def test_spider_itergen_uses_the_checked_in_upstream_search_settings():
    kwargs = _itergen_generation_kwargs(
        dataset="spider",
        max_tokens=8192,
        max_new_tokens=176,
    )

    assert kwargs["do_sample"] is False
    assert kwargs["recurrence_penalty"] == 0.3


def test_spider_itergen_advances_by_schema_units_and_backtracks_invalid_names():
    class FakeIterGen:
        def __init__(self):
            self.started_with = None
            self.forward_calls = []
            self.backward_calls = []
            self.iteration = 0

        def start(self, prompt):
            self.started_with = prompt

        def finished(self):
            return self.iteration >= 2

        def forward(self, **kwargs):
            self.forward_calls.append(kwargs)
            self.iteration += 1
            return ["SELECT invented FROM singer" if self.iteration == 1 else "SELECT name FROM singer"]

        def view(self, unit):
            if unit == "column_name":
                return [["invented"]] if self.iteration == 1 else [["name"]]
            if unit == "table_name":
                return [["singer"]]
            raise AssertionError(unit)

        def backward(self, unit):
            self.backward_calls.append(unit)

    value = FakeIterGen()
    result = _itergen_generate_spider(
        value,
        "SQL:",
        {
            "db_info": "# singer ( singer_id , name )",
        },
    )

    assert value.started_with == "SQL:"
    assert value.forward_calls == [
        {"units": ["column_name", "table_name"], "num": 1},
        {"units": ["column_name", "table_name"], "num": 1},
    ]
    assert value.backward_calls == ["column_name"]
    assert result == "SELECT name FROM singer"


@pytest.mark.parametrize("strategy", ["gcd", "itergen"])
def test_smiles_fixed_decoders_honor_the_requested_token_budget(strategy):
    assert _legacy_fixed_max_new_tokens("smiles", 400, strategy=strategy) == 400


def test_crane_smiles_starts_constrained_without_visible_delimiters():
    surface = _crane_adaptive_surface("smiles", "start: /[A-Z]+/")

    assert surface == {
        "grammar": "start: /[A-Z]+/",
        "start_symbol": "",
        "start_in_grammar": False,
        "end_symbol": None,
        "end_in_grammar": False,
        "start_inside_constrained": True,
    }
