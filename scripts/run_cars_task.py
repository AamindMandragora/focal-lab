#!/usr/bin/env python3
"""Run one CARS task with explicit files and output directory."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def _torch_dtype_for_device(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def _add_cars_paths(cars_repo: Path) -> None:
    repo = cars_repo.expanduser().resolve()
    path_str = str(repo)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cars-repo", type=Path, required=True)
    parser.add_argument("--grammar-file", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--sample-style", type=str, default="cars")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--target-samples", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.target_samples <= 0:
        raise SystemExit("--target-samples must be > 0")
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be > 0")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be > 0")

    _add_cars_paths(args.cars_repo)

    import cars
    import cars.lib
    import mcmc
    import mcmc.lib
    from transformers import GenerationConfig
    from transformers.generation.logits_process import InfNanRemoveLogitsProcessor, LogitsProcessorList

    class CompatibleConstrainedModel(cars.lib.ConstrainedModel):
        def __init__(self, model_id: str, grammar_str: str | None = None, **kwargs):
            self.model_id = model_id
            self.tokenizer = cars.lib.AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = cars.lib.AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                **kwargs,
            )
            self.model.eval()
            if model_id == "hsultanbey/codegen350multi_finetuned":
                self.model.resize_token_embeddings(len(self.tokenizer))
            if grammar_str is not None:
                self._set_grammar_constraint(grammar_str)

        def _format_prompt(self, prompt: str) -> str:
            if self.model_id in self.HF_BASE_MODELS:
                return prompt
            chat_template = getattr(self.tokenizer, "chat_template", None)
            if chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(formatted, str):
                    return formatted
            return prompt

        def _generate(self, input_ids: torch.Tensor, max_new_tokens: int):
            generation_config = GenerationConfig(
                max_new_tokens=int(max_new_tokens),
                num_return_sequences=1,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            self.gcd_logits_processor.reset()
            logits_processor_list = LogitsProcessorList([self.gcd_logits_processor, InfNanRemoveLogitsProcessor()])
            output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor_list,
            )
            output_ids = output.sequences
            raw_logprob = self.gcd_logits_processor.generation_ended(output_ids)
            output_ids = output_ids[:, input_ids.shape[1]:]
            output_scores = torch.stack(output.scores, dim=1)
            if output_ids.shape[1] != output_scores.shape[1]:
                raise RuntimeError(
                    f"Output ids/scores length mismatch: {output_ids.shape[1]} vs {output_scores.shape[1]}"
                )
            return output_ids, output_scores, float(raw_logprob.item())

    grammar_text = args.grammar_file.read_text()
    prompt_text = args.prompt_file.read_text()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    dtype = _torch_dtype_for_device(args.device)
    previous_cwd = Path.cwd()
    os.chdir(args.cars_repo.expanduser().resolve())
    try:
        model = CompatibleConstrainedModel(args.model_name, grammar_text, torch_dtype=dtype)
    finally:
        os.chdir(previous_cwd)

    if args.sample_style in cars.all_sample_styles():
        runner = cars.CARS(
            model=model,
            prompt=prompt_text,
            sample_style=args.sample_style,
            log_dir=str(args.log_dir),
        )
        runner.get_samples(
            n_samples=1,
            n_steps=int(args.n_steps),
            stop_after=int(args.target_samples),
            max_new_tokens=int(args.max_new_tokens),
        )
        return 0

    if args.sample_style in mcmc.all_sample_styles():
        runner = mcmc.MCMC(
            model=model,
            prompt=prompt_text,
            propose_style=args.sample_style,
            log_dir=str(args.log_dir),
        )
        runner.get_samples(
            n_samples=int(args.target_samples),
            n_steps=int(args.n_steps),
            max_new_tokens=int(args.max_new_tokens),
        )
        return 0

    available = cars.all_sample_styles() + mcmc.all_sample_styles()
    raise SystemExit(f"Unknown sample style {args.sample_style!r}. Available: {available}")


if __name__ == "__main__":
    raise SystemExit(main())
