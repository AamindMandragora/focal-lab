module CRANE {
  import opened Decoding
  import opened CSD

  // Whole-loop CSD execution wrapper (this is the new meaning of “CSD”).
  function CRANE_RunCSD(program: Program, prompt: Prefix, maxSteps: nat): Prefix
  {
    Run(program, prompt, maxSteps)
  }

  predicate IsCorrectCSDProgram(
    program: Program,
    prompt: Prefix,
    maxSteps: nat,
    result: Prefix
  )
  {
    result == CRANE_RunCSD(program, prompt, maxSteps)
    && |result| >= |prompt|
    && prompt == result[..|prompt|]
    && |result| <= |prompt| + maxSteps
    // If the program returns from the constrained fallback, ParseOk must hold.
    // If it returns from unconstrained retries, ParseOk may or may not hold
    // depending on the chosen acceptance predicate; the definition of Run uses
    // ParseOk to decide acceptance.
  }

  lemma CRANE_RunCSD_Correct(program: Program, prompt: Prefix, maxSteps: nat)
    ensures IsCorrectCSDProgram(program, prompt, maxSteps, CRANE_RunCSD(program, prompt, maxSteps))
  {
  }
}