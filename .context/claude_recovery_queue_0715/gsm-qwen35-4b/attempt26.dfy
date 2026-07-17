// CSD_RATIONALE_BEGIN
// Primary fix: Attempt 25 used `next == "<<"` for delimiter detection, causing
// constrained mode to activate in only 7/49 examples (the token may be " <<" or
// split). Replacing with `RenderedEndsWith(generated, "<<")` should restore
// constrained activity to ~42/49 examples, directly fixing the syntax failures.
//
// Secondary fix: Hard ConstrainedStep inside every span guarantees the parser
// rejects `{n}`, `{frac_1}`, `$`, `**` etc., eliminating the 8 final_span_invalid
// examples from syntax failures.
//
// Tertiary fix: Force `OpenConstrainedSpan` at step forceThreshold if no span
// has ever been opened (fixes 6 no-span / mode_C examples that hit max steps
// without any `<<`).
//
// Post-loop: `CloseSpanWithinBudget` on remaining budget if span still open
// (fixes 6 span-opened-but-never-closed examples from prior feedback).
//
// Guidance updated to explicitly ban curly braces inside expressions and explain
// bare variable names, since the primary failure pattern was `{n}` template syntax.
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   - Init: satisfied by method precondition.
//   - Force-open branch: OpenConstrainedSpan sets currentConstrainedOut := [];
//     parser.IsValidPrefix([]) holds by method precondition.
//   - Unconstrained step with RenderedEndsWith match: sets currentConstrainedOut := [];
//     same argument.
//   - Constrained, closed == true: CloseSpanIfComplete postcondition ensures
//     !insideConstrainedOut, making the implication vacuously true.
//   - Constrained, closed == false, valid token appended: AppendConstrainedToken
//     postcondition preserves parser.IsValidPrefix(currentConstrainedOut).
//   - Phase 2 CloseSpanWithinBudget: postcondition directly ensures
//     insideOut ==> parser.IsValidPrefix(currentOut).
//
// progress: |generated| <= |generatedPrefix| + steps
//   - Init: 0 <= 0. ✓
//   - Force-open branch: OpenConstrainedSpan appends one << token (+1 length),
//     steps += 1 (+1 steps). ✓
//   - Unconstrained step: UnconstrainedStep generates one token; EOS breaks without
//     appending; otherwise generated grows by 1, steps grows by 1. ✓
//   - Constrained, closed == true: CloseSpanIfComplete appends >> (cost +1 token-step,
//     +1 length), steps += 1. ✓
//   - Constrained, closed == false: CloseSpanIfComplete is no-op (+0 length, cost 0),
//     steps += 1; ConstrainedStep samples one token (cost 1, no visible append);
//     AppendConstrainedToken adds at most 1 visible token. Net: length +1, steps +1. ✓
//   - Phase 2: CloseSpanWithinBudget postcondition: |cg| <= |generated| + closeBudget
//     = |generated| + (maxSteps - steps_pre) <= |generatedPrefix| + maxSteps. ✓
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve step by step. Wrap each intermediate result and the final answer in <<expr>>. CRITICAL inside << >>: bare variable names with NO curly braces. Write n not {n}, frac_1 not {frac_1}, mult not {mult}. Allowed: variable names, numbers, +, -, *, /, //, %, (, ), int(). Use // for integer division. Wrap whole-number results in int(). No LaTeX, no {}, no $, no **. Each answer: one <<expr>>.");

var steps: nat := 0;
var spanEverOpened: bool := insideConstrained;
var forceThreshold: nat := if maxSteps > 200 then maxSteps - 200 else 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  if !insideConstrainedOut {
    if steps >= forceThreshold && !spanEverOpened {
      // Model never opened a span; force one to guarantee an answer
      var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
      generated := og;
      insideConstrainedOut := oi;
      currentConstrainedOut := oc;
      steps := steps + 1;
      spanEverOpened := true;
    } else {
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next == eosToken {
        break;
      } else {
        generated := generated + [next];
        if RenderedEndsWith(generated, "<<") {
          insideConstrainedOut := true;
          currentConstrainedOut := [];
          spanEverOpened := true;
        }
      }
    }
  } else {
    // Inside constrained span: attempt close, then hard-constrained step
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    if closed {
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
    } else {
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
      if next == eosToken {
        break;
      } else {
        var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);
        if valid {
          var ag, ai, ac := helpers.AppendConstrainedToken(
            lm, parser, generated, currentConstrainedOut, next
          );
          generated := ag;
          insideConstrainedOut := ai;
          currentConstrainedOut := ac;
        }
      }
    }
  }
}

// Phase 2: if budget remains and span is still open, close it reliably
if insideConstrainedOut && steps < maxSteps {
  var closeBudget := maxSteps - steps;
  var cg, ci, cc := helpers.CloseSpanWithinBudget(
    lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
  );
  generated := cg;
  insideConstrainedOut := ci;
  currentConstrainedOut := cc;
  steps := maxSteps;
}

cost := steps;
