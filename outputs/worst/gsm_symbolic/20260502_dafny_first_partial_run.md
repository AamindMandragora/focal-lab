Task family: gsm_symbolic.
Dafny-first fallback path now verifies, compiles, and runs, but one 20-example partial evaluation exposed a remaining policy failure mode.
Observed good behavior: examples 1, 2, and 4 produced clean late spans such as `<<16 * 8.5 + 4 * 10.5 + 13>>`, `<<1/2 * 120 = 60>>`, and `<<15 * 20 + 5 * 20 * 0.75>>`.
Observed bad behavior: one example consumed the full 900-step budget and produced a long repetitive prose ramble with no useful final constrained span.
Avoid Dafny fallback strategies that only open on one late counter schedule with weak semantic cues; they can still miss the final span and drift until the step cap.
Preferred next fix: strengthen close/open intent with clearer answer cues, more aggressive transition out of prose under budget pressure, or explicit scratch-to-final state so hard examples do not stay in unconstrained narration for hundreds of tokens.
