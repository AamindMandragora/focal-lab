# Generated Artifacts

This directory stores synthesis run outputs.

## Per-Run Structure

Each generated CSD run should create a dedicated folder:

- `generated/<output_name>_<timestamp>_<token>/`
  - `dafny/`: generated Dafny source snapshots.
  - `python/`: compiled Python modules from Dafny build.
  - `results/`: success/failure reports and evaluation outputs.

This structure keeps source, executable artifacts, and results co-located and easy to inspect.
