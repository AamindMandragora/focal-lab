// CSD_RATIONALE_BEGIN
// Diagnosis: syntax is 90% (goal 96%), accuracy is 70% (goal 45%).
// The core problem: only 1/50 examples actually used constrained mode.
// The model emits "<<" as two separate "<" tokens, so our trigger fires but
// OpenConstrainedSpan appends another "<<", creating "<<<" or doubling delimiters.
// Actually looking at the data: 50/50 have visible "<<", but only 1/50 used
// constrained branch. The previous attempt triggered on "<" and called
// OpenConstrainedSpan, but OpenConstrainedSpan appends "<<" while the model's
// "<" was discarded — yet the model might have already emitted the first "<"
// and we're calling OpenConstrainedSpan which adds "<<", giving "<<<".
//
// The real issue: the model naturally produces well-formed "<<" spans in 45/50
// cases (syntax 90%). The 5 malformed cases have "if...else", comparisons, etc.
// inside spans. These are hard to fix without active constrained generation.
//
// Strategy: FORCE the entire span structure proactively.
// Instead of detecting when the model wants a span, we:
// 1. Generate N unconstrained tokens (free reasoning)
// 2. Force OpenConstrainedSpan (appends "<<", cost 1)
// 3. Drive span content with ConstrainedStep until complete
// 4. Force CloseConstrainedSpan (appends ">>", cost 1)
// 5. Repeat
//
// But this is too rigid — we don't know where spans should go.
//
// Better: The model's unconstrained output is already 90% syntactically valid.
// The problem is 5 malformed spans. The approach of triggering on "<" worked
// for 1/50 — but broke others (accuracy went from 64% to 70% but syntax stayed 90%).
//
// Key insight: The model is doing well on its own (90% syntax, 70% accuracy).
// We need to ONLY fix the 5 malformed span cases without breaking the 45 good ones.
// The malformed content has "if...else", "min(...)", "int(...)", conditional expressions.
//
// The challenge: we can't selectively apply constraints to only the bad cases.
// We need a strategy that works for all cases.
//
// Approach: Generate EVERYTHING unconstrained, but after the model emits "<<",
// we detect it and retroactively enter constrained mode for the content.
// The problem was that "<<" doesn't come as a single token.
//
// What if we check for "<" as the first character of the token?
// In Dafny we can check next[0] == '<' if |next| > 0.
// But Dafny strings... Token is `string`, so we can check |next| >= 1 && next[0] == '<'.
//
// Strategy: when we see a token starting with "<", we call OpenConstrainedSpan
// to force "<<" and enter constrained mode. The model's "<" or "<<" token is
// discarded. This is the same as attempt 5 but with a broader trigger.
//
// Wait — attempt 5 already tried next == "<" || next == "<<" and got 1/50.
// So the model is NOT emitting single "<" tokens either!
//
// Let me reconsider. The model emits "<<" as part of its output and it shows up
// in the concatenated text. But what token IS it? Looking at Qwen tokenizer:
// "<<" is tokenized as a single token "<<" in most cases.
// But our check next == "<<" in attempt 4 only got 3/50.
// And next == "<" || next == "<<" in attempt 5 got 1/50.
//
// This is very strange. The model IS emitting "<<" (50/50 have it), but our
// trigger fires only 1-3 times. This means the model emits "<<" INSIDE a longer
// token! E.g., the token might be "<<n" or " <<" (with space) or "<<x".
//
// Qwen tokenizer: " <<" (space + <<) is likely a single token. Let me check:
// The token " <<" (with leading space) is common in Qwen tokenizations.
// So `next == " <<"` might be the right check!
//
// Also possible: the model emits "<<" as part of a longer token like "<<n1".
//
// Since we can't do substring matching easily, let's try a different approach:
// check multiple possible token forms that contain "<<".
//
// In Dafny, we can check: next == "<<" || next == " <<" || (|next| >= 2 && next[0] == '<' && next[1] == '<')
//
// The last check handles any token starting with "<<".
//
// For tokens that START with "<<" but have more characters (like "<<n"),
// we discard the whole token and let OpenConstrainedSpan add "<<".
// The extra characters after "<<" are lost, but that's OK since the constrained
// step will regenerate them correctly.
//
// For " <<" (with space): we should append the space first, then enter span.
// But for simplicity, we can just trigger OpenConstrainedSpan and lose the space.
// The model will regenerate the span content correctly.
//
// This approach should dramatically increase constrained branch engagement.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   1. Outside span, non-trigger token: insideConstrainedOut stays false, implication vacuously true.
//   2. Outside span, trigger token (starts with "<<"): we call OpenConstrainedSpan which sets
//      insideConstrainedOut := true and currentConstrainedOut := [].
//      parser.IsValidPrefix([]) holds by precondition. Invariant preserved.
//   3. CloseConstrainedSpan branch: sets insideConstrainedOut := false, making
//      the implication vacuously true. Invariant preserved.
//   4. Inside span, ConstrainedStep branch: ConstrainedStep hard-masks to
//      parser-valid tokens plus EOS. AppendConstrainedToken appends next only
//      when next != eosToken, and ConstrainedStep guarantees parser validity, so
//      parser.IsValidPrefix(currentConstrainedOut + [next]) holds. Invariant preserved.
//
// progress:
//   1. Outside span, non-trigger: steps += 1, generated grows by 1. Bound preserved.
//   2. Outside span, trigger: steps += 1 (UnconstrainedStep), then if budget remains,
//      OpenConstrainedSpan costs 1 step (steps += 1). generated grows by 1 ("<<").
//      We do NOT append next, so |generated| grows by at most 1 while steps grows by 2.
//      Bound |generated| <= |generatedPrefix| + steps preserved.
//   3. CloseConstrainedSpan: steps += 1, generated grows by at most 1 (">>"). Bound preserved.
//   4. Inside span, ConstrainedStep: steps += 1, generated grows by at most 1. Bound preserved.
//   5. EOS break: loop exits, no further growth.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve the math word problem step by step. Use << >> delimiters to wrap every arithmetic expression and the final numeric answer. Inside << >>, use only: variable names, numbers, +, -, *, /, (, ). Do NOT use: =, if, else, min, max, int(), ceil(), floor(), or any Python functions. Keep expressions simple and direct. Final answer line: #### <<expression>>.");

var steps: nat := 0;

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
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      // Check if this token starts with "<<" (handles "<<", " <<", "<<x", etc.)
      var startsWithDoubleAngle := |next| >= 2 && next[0] == '<' && next[1] == '<';
      var isSpaceDoubleAngle := |next| >= 3 && next[0] == ' ' && next[1] == '<' && next[2] == '<';
      if startsWithDoubleAngle || isSpaceDoubleAngle {
        // Model wants to open a span — force OpenConstrainedSpan
        // If token starts with " <<", append the space first
        if isSpaceDoubleAngle && !startsWithDoubleAngle {
          generated := generated + [" "];
        }
        if steps < maxSteps {
          var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
          generated := openedGenerated;
          insideConstrainedOut := openedInside;
          currentConstrainedOut := openedCurrent;
          steps := steps + 1;
        }
      } else {
        generated := generated + [next];
      }
    }
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.ConstrainedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, eosToken
    );
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := appendedGenerated;
      insideConstrainedOut := appendedInside;
      currentConstrainedOut := appendedCurrent;
    }
  }
}

cost := steps;