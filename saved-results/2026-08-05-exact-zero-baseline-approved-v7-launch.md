# Exact-zero baseline approved-v7 launch

Date: 2026-08-05

## Purpose

This record freezes the 15 unresolved exact-zero source artifacts before their approved replacements run. Every source scored 0.0 accuracy and 0.0 syntax and remains preserved. Replacements write only under `outputs/baselines/exact-zero-repair-20260805-approved-v7`.

Repair branch: `codex/full-baseline-campaign-20260803`

Repair commit: `bc1d9f3bf08b1aaf4a726d94bfd79eeebde15ea6`

Synthesis launch remains blocked by `.context/exact-zero-repair-synthesis.blocked`.

## Frozen sources

| Label | Source SHA-256 |
|---|---|
| `smiles-acrylates-qwen25-1p5b-itergen` | `8f225e622654845896f050714b8d04712ac434fb03a59c14b1638b845432f47a` |
| `smiles-acrylates-qwen25-7b-crane` | `e553157664508922bd37426d2a57eb403690bbe86665d67be6bebf317f1ec665` |
| `smiles-acrylates-qwen35-2b-itergen` | `0fdc4d6b122fc70e20a44d82307e126bebb1e766dedc4481b5f2c6b912ac3683` |
| `smiles-acrylates-qwen35-4b-itergen` | `f3e09eec2650f772b56f6008d11cba44a43997c7127dfb8754346be5a9aa58a6` |
| `smiles-chain_extenders-qwen25-7b-itergen` | `ea44c22be59d529f125a433b0d87b5092208a09122898bc10f774f4b659dde54` |
| `smiles-chain_extenders-qwen35-2b-itergen` | `04782852612fcd293dca3010aeb824672df05cd33b7351d693fdd9774a11524e` |
| `smiles-chain_extenders-qwen35-4b-itergen` | `e8bf31cbd536c7420a139cd03ed97752c50b8a0d99a09cb8dfd456cdebcef331` |
| `smiles-isocyanates-qwen25-1p5b-crane` | `c32d357eb5ad0f29e989a1074c23276d75fc33bb79b1e22aaaccb9fed532cc7b` |
| `smiles-isocyanates-qwen25-1p5b-itergen` | `51e61984969382dbf84dc3471c265366eb552248ddf43c60f754f75783090a2f` |
| `smiles-isocyanates-qwen25-7b-crane` | `fd18f7c89495a330ecc576f6218d2bd14b874a2a883179ee699e08a8a0afd9c7` |
| `smiles-isocyanates-qwen25-7b-itergen` | `c79ef2890be2d66b0e7fd5fe0abeb7cd97b18914983b2da77f414ccfd548bd21` |
| `smiles-isocyanates-qwen35-2b-itergen` | `42935b2824c19e8438f8bb4ba69d65222b00a5d97e9507910fca41170e0bdf1c` |
| `smiles-isocyanates-qwen35-4b-itergen` | `abc1e3974c182e7739d6bac8ba26da2778a61be4a37762b498a374c7fe31ea40` |
| `spider-qwen35-2b-itergen` | `22c2932d77c656a687e8f5490fc710b8e663d150cf3ddae7827295e798499cca` |
| `spider-qwen35-4b-itergen` | `1affdc1ea99b3eabf23890846be4dfb67114c73149d252a90672ae0e72a2ea91` |

Source paths follow `outputs/baselines/exact-zero-repair-20260804/<cohort>/<model>/<strategy>.json`; replacement paths use the same suffix under the approved-v7 root.

## Pre-launch smoke evidence

| Path | SHA-256 | Result |
|---|---|---|
| `outputs/baselines/exact-zero-repair-20260805-approved-v4-smoke/smiles-acrylates-qwen25-7b-crane.json` | `afb54bf0bb59c29ea018fa285433dcc4bd9a9f30c0948716d0b94df7de9db6fe` | One valid 27-character inner SMILES; accuracy 1.0, syntax 1.0. |
| `outputs/baselines/exact-zero-repair-20260805-approved-v4-smoke/smiles-acrylates-qwen25-1p5b-itergen.json` | `6315fd3b01e71bcc030744badb8ca358d473e46b9c704d2156cb88ec0471cfc3` | Sampling path produced one nonblank 402-character output; malformed, so full cells remain under skeptical review. |
| `outputs/baselines/exact-zero-repair-20260805-approved-v6-smoke/spider-qwen35-2b-itergen.json` | `84b094bca2032010b984a7a62c78299a730b706df4a68fed89d2759921fcf3fd` | Chat-rendered path produced one nonblank 558-character output instead of whitespace; wrong SQL, so full cells remain under skeptical review. |

The failed approved-v5 Spider invocation is preserved in its log; it used a nonexistent split path and produced no baseline JSON.

## Reproduce

Run `scripts/run_focal_collection_pool.py` with campaign `full-baseline-20260803`, campaign output name `exact-zero-repair-20260805-approved-v7`, GPUs `0,2,3`, and exactly the 15 labels above. Do not remove the synthesis block until all replacements have passed SHA-bound skeptical review and corrected evidence has been rebuilt.

## 09:40 UTC progress and cache diagnosis

Five v7 cells had finished with exit code 0:

| Label | Replacement SHA-256 | Accuracy | Syntax | Review state |
|---|---|---:|---:|---|
| `smiles-acrylates-qwen25-1p5b-itergen` | `6e93496403f9b59fd9d9a4f2ae424f07518258546d1942fa863c1de2dae860fe` | 0.00 | 0.04 | Functioning, poor baseline; 50 nonblank and 49 unique outputs. |
| `smiles-acrylates-qwen25-7b-crane` | `4f458073f3abc8d2061cd76eee5e3e95f4a7ee5759bb8f15ecf835849325c03d` | 0.06 | 0.86 | Accepted. |
| `spider-qwen35-2b-itergen` | `6c1aa6c1b9cdbed7ccf9089e454632aee33f0b48b1598ca3dd26fc40a8b22250` | 0.00 | 0.00 | Quarantined: 300 copies of one 558-character non-SQL answer. |
| `smiles-acrylates-qwen35-2b-itergen` | `88224dade3ee84355860a5593a658237b305ce5774fe23aa2fcb2eeb21c1b111` | 0.00 | 0.00 | Quarantined: started before the prompt-cache repair. |
| `smiles-chain_extenders-qwen25-7b-itergen` | `c80305af134eefca76f86b00adfaf9f8ffb5d1129d46199be1b1918b68bf807e` | 0.86 | 0.86 | Accepted. |

The Qwen3.5 cache probe showed that `DynamicCache(config=...)` is truthy before
it contains tokens: the 2B and 4B models allocate 24 and 32 empty layers while
`get_seq_length()` remains zero. Legacy IterGen used cache truthiness and thus
sent only the final prompt token on the first forward pass. The narrow repair
uses `get_seq_length() > 0`; its regression tests cover truthy empty and
populated caches, and the tracked legacy patch is
`environment/legacy_patches/itergen/011-empty-config-cache-full-prompt.patch`.

Verification after the repair:

- `tests/test_itergen_transformers_compat.py`: 19 passed.
- Nested IterGen prompt-cache and recurrence tests: 3 passed.
- A post-repair Qwen3.5-2B chain-extenders worker started at 09:36:17 UTC and
  produced a nonblank, chain-extender-specific 576-character first output with
  parser progress. This validates that the full prompt reaches live decoding;
  the complete cell remains under skeptical score review.

The two Spider Qwen3.5 and two acrylates Qwen3.5 v7 jobs began before the
cache repair. Preserve and quarantine all four, then rerun only those labels in
a new versioned root after v7 finishes. Synthesis remains blocked.

## Post-repair Qwen3.5 live results

The first two complete Qwen3.5 cells that started after the cache repair both
produced functioning, nonzero baselines:

| Label | Replacement SHA-256 | Accuracy | Syntax | Structure |
|---|---|---:|---:|---|
| `smiles-chain_extenders-qwen35-2b-itergen` | `8f773ee079a362b46b8e61c6949d1419298ac1a14b1e68b6815c264bbeb5d9c6` | 0.34 | 0.94 | 50/50 nonblank, 10 unique outputs |
| `smiles-chain_extenders-qwen35-4b-itergen` | `6845e5cc9293e40b9cc889396428727defbc69ede0e0508e9b1f2a70a1e302e5` | 0.80 | 0.82 | 50/50 nonblank, 16 unique outputs |

These results are accepted evidence because they have complete row counts,
diverse nonblank outputs, parser progress, and nonzero scores. They also give
live causal confirmation that checking cached sequence length restored the
Qwen3.5 prompt path. They do not make either pre-repair acrylates artifact
valid; those remain quarantined for v8 replacement.

## SMILES CRANE isocyanates results

Both repaired CRANE cells produced complete, varied, nonzero evidence:

| Label | Replacement SHA-256 | Accuracy | Syntax | Structure |
|---|---|---:|---:|---|
| `smiles-isocyanates-qwen25-1p5b-crane` | `6295bd786e1102c01e23fe92a070364cfc21ad017bace957c3bb275a4f2b3043` | 0.18 | 0.28 | 22/50 nonblank, 19 unique outputs |
| `smiles-isocyanates-qwen25-7b-crane` | `67dc4e922a66158ebd5e7c5a102e75a798525c246fd77f7510faa04ee973febe` | 0.68 | 0.76 | 39/50 nonblank, 11 unique outputs |

Blank rows remain syntax failures, but they do not invalidate either complete
cell: both have diverse parsed outputs and nonzero accuracy. The per-example
parser exception in the 1.5B log is reflected by failed rows, not a lost run.
