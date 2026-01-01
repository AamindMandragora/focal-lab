// GENERATED FILE (template).
// This file will be overwritten by scripts/synthesize_csd.py.
//
// Qwen must ONLY fill in the body of GeneratedProgram() with a single Dafny
// expression of type CSD.Program. Do not add new modules/functions/lemmas.

module GeneratedCSD {
  import opened CSD
  import opened Decoding

  // Qwen output goes inside the braces.
  function GeneratedProgram(): Program
  {
    // QWEN_SNIPPET_START
    // Example placeholder: always fall back to constrained generation.
    ReturnParsed(0, MaskWithSynCode(WithRejection(Base(ProposalStrategy.Temperature))))
    // QWEN_SNIPPET_END
  }
}


