# SMILES Data Assets

This directory holds class-specific assets used by the SMILES benchmark implementation.

## Asset Types

- `.lark` files: grammar definitions for each molecule class.
- `.txt` files: class exemplars and data snippets used during prompting/evaluation.

## Current Classes

- `acrylates`
- `chain_extenders`
- `isocyanates`

## Usage

The SMILES dataset/benchmark modules load these assets to construct class-aware prompts and parser constraints so strategies are evaluated against class-specific molecular syntax expectations.
