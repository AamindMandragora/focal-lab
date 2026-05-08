# SMILES Data Assets

This directory holds class-specific assets used by the SMILES benchmark implementation.

## Asset Types

- `.txt` files: class exemplars and data snippets used during prompting/evaluation.

Class grammars live in `synthesis/evaluate/grammars/` as:
- `smiles_acrylates.lark`
- `smiles_chain_extenders.lark`
- `smiles_isocyanates.lark`

## Current Classes

- `acrylates`
- `chain_extenders`
- `isocyanates`

## Usage

The SMILES dataset/benchmark modules load these assets to construct class-aware prompts and parser constraints so strategies are evaluated against class-specific molecular syntax expectations.
