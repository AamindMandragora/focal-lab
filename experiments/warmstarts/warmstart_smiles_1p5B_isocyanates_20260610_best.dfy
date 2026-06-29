// CSD_RATIONALE_BEGIN
// The core problem: the model generates "O" as the first SMILES token, which is
// a complete SMILES (just oxygen atom). CloseSpanIfComplete immediately closes
// the span after 1 token, giving <<O>> which is valid SMILES but NOT an isocyanate.
//
// The fundamental issue is that the SMILES parser accepts single atoms as complete
// molecules. We need to force the model to generate isocyanate-specific tokens
// BEFORE the parser can declare completion.
//
// Key insight from the failure trace:
// - "entered_constrained_mode_too_early": spans open at token 0
// - "tiny_span_dominant": spans are length 1
// - The model generates "O" which satisfies the SMILES grammar immediately
//
// Fix strategy:
// 1. Use ConstrainedGeneration which runs until parser completion OR EOS OR budget
//    This is the cleanest approach - it handles all the complexity internally
// 2. Don't use CloseSpanIfComplete in a tight loop - it closes too eagerly on single atoms
// 3. The key is: use ConstrainedGeneration which internally loops ConstrainedStep
//    until the parser says complete, then we just close the span.
//
// But wait - ConstrainedGeneration starts from empty currentConstrained, so it
// builds the full SMILES from scratch. The problem is that "O" IS a complete
// SMILES. The parser will terminate there.
//
// The real fix: we need to FORCE the first few tokens to be isocyanate-specific.
// We can use IsTokenValidNext + AppendConstrainedToken to force specific tokens
// like "N", "=", "C", "=", "O" at the start, then let the model continue freely.
//
// Actually, looking at the guidance - the model is picking "O" because it sees
// "O=C=N..." type patterns. With guidance saying "N=C=O group", the model should
// pick "N" or a carbon first.
//
// Better approach: Use AdaptiveConstrainedStepWithPenalties with penalties on
// single-atom-complete tokens, combined with stronger guidance, and use
// RollbackAndContinue if we detect a tiny span.
//
// SIMPLEST CORRECT APPROACH:
// 1. Open span
// 2. Check if "N" is valid next - if so, force "N" as first token (or try to
//    force isocyanate scaffold tokens)
// 3. Use AdaptiveConstrainedStep for subsequent tokens
// 4. Only call CloseSpanIfComplete after sufficient length (>= 5 tokens)
// 5. At end, if still inside and parser is complete, close; if not complete, rollback
//
// The verification issue from attempt 14: AppendConstrainedToken requires
// !parser.IsCompletePrefix(currentConstrained). We must check this before calling.
//
// REVISED PLAN:
// - Force first token to be valid non-terminating isocyanate start
//   by using penalty on eosToken and using AdaptiveConstrainedStep
// - After >= 5 tokens, check CloseSpanIfComplete
// - The key insight: we need stronger guidance AND we need to avoid the loop
//   closing after 1 token
//
// Wait - let me re-read. The issue in attempt 14 (best result) is:
// Failure mode: entered_constrained_mode_too_early + tiny spans
// This means the span opens, generates 1 token (O), closes (because CloseSpanIfComplete
// fired), and that's the answer.
//
// In the BEST RESULT code: CloseSpanIfComplete is called FIRST in every iteration
// of the while loop. So on the FIRST iteration (currentConstrainedOut = []), it
// checks if [] is a complete SMILES prefix. Is empty string a complete SMILES? 
// Probably not! So it should return closed=false. Then it generates a token.
// If the token is "O" and "O" is appended, then on the NEXT iteration
// CloseSpanIfComplete is called with ["O"] and "O" IS a complete SMILES, so it closes.
// That gives a 1-token span <<O>>.
//
// The fix: don't call CloseSpanIfComplete until |currentConstrainedOut| >= minLen.
// We tried this in attempt 14 but added verification issues.
//
// NEW PLAN: The simplest fix that avoids verification issues:
// 1. Use a phase counter (phaseTokens). First generate N tokens WITHOUT checking
//    CloseSpanIfComplete. Only check close after phaseTokens generated.
// 2. For the forced phase, use AdaptiveConstrainedStep + check !IsCompletePrefix
//    before AppendConstrainedToken.
// 3. After phase complete, use the standard CloseSpanIfComplete loop.
// 4. Guidance should strongly push toward "O=C=N" or "N=C=O" patterns.
//
// The verification fix for AppendConstrainedToken: add the check
// !parser.IsCompletePrefix(currentConstrainedOut) before calling it.
// If it IS complete, we should close instead.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   - OpenConstrainedSpan: sets currentConstrainedOut := [], parser.IsValidPrefix([]) by precondition.
//   - Phase 1 (building tokens without close check): AdaptiveConstrainedStep returns a
//     parser-valid token or eosToken. We check !parser.IsCompletePrefix before calling
//     AppendConstrainedToken, satisfying its precondition. AppendConstrainedToken preserves
//     parser.IsValidPrefix(currentConstrainedOut + [next]).
//     If parser.IsCompletePrefix is true (unexpected), we call CloseConstrainedSpan which
//     sets insideConstrainedOut=false, making the implication vacuous.
//   - Phase 2 (CloseSpanIfComplete loop): closed=true sets insideConstrainedOut=false
//     (implication vacuous); closed=false leaves state unchanged (invariant preserved by
//     induction). AdaptiveConstrainedStep + AppendConstrainedToken with IsCompletePrefix
//     guard preserve validity.
//   - CloseConstrainedSpan (final): sets insideConstrainedOut=false, implication vacuous.
//
// progress: |generated| <= |generatedPrefix| + steps
//   - OpenConstrainedSpan: steps += 1, |generated| += 1. Balance preserved.
//   - Phase 1 tokens: each AdaptiveConstrainedStep costs 1 step; AppendConstrainedToken
//     adds 1 token. So |generated| grows by at most 1 per step. Preserved.
//     CloseConstrainedSpan (if IsComplete in phase 1): steps += 1, |generated| += 1.
//   - CloseSpanIfComplete: steps += 1, |generated| grows by at most 1. Preserved.
//   - Phase 2 AdaptiveConstrainedStep: steps += 1, AppendConstrainedToken adds 1 token.
//   - Final CloseConstrainedSpan: steps += 1, |generated| += 1. Preserved.
//   - EOS: steps already incremented, no append. Preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Task: output exactly one SMILES string for an isocyanate molecule. Isocyanates must contain the N=C=O functional group. Start the SMILES with the isocyanate nitrogen: begin with O=C=N and then add a substituent R group (alkyl or aryl). Valid examples: O=C=NC, O=C=NCC, O=C=NCCCl, O=C=Nc1ccccc1. Do NOT output a single atom. Output at least 5 atoms.");

var steps: nat := 0;
var minPhaseTokens: nat := 5;  // Generate at least this many tokens before allowing close

// Open the constrained span immediately
if !insideConstrainedOut && steps < maxSteps {
  var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
  generated := og;
  insideConstrainedOut := oi;
  currentConstrainedOut := oc;
  steps := steps + 1;
}

// Phase 1: Generate at least minPhaseTokens tokens without checking completion
var phaseCount: nat := 0;
while steps < maxSteps && insideConstrainedOut && phaseCount < minPhaseTokens
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  var isComp := parser.IsCompletePrefix(currentConstrainedOut);
  if isComp {
    // Parser already complete - close and exit phase 1
    if steps < maxSteps {
      var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := steps + 1;
    }
    break;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.AdaptiveConstrainedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, 20, eosToken
    );
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      var isCompCheck := parser.IsCompletePrefix(currentConstrainedOut);
      if !isCompCheck {
        var ag, ai, ac := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := ag;
        insideConstrainedOut := ai;
        currentConstrainedOut := ac;
        phaseCount := phaseCount + 1;
      } else {
        // Became complete after check - close
        if steps < maxSteps {
          var cg2, ci2, cc2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
          generated := cg2;
          insideConstrainedOut := ci2;
          currentConstrainedOut := cc2;
          steps := steps + 1;
        }
        break;
      }
    }
  }
}

// Phase 2: Continue generation with CloseSpanIfComplete check
while steps < maxSteps && insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  // Try to close if complete
  var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
  steps := steps + 1;
  if closed {
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    break;
  } else {
    // Generate next token
    if steps < maxSteps {
      var isComp2 := parser.IsCompletePrefix(currentConstrainedOut);
      if isComp2 {
        // Shouldn't happen (CloseSpanIfComplete would have closed), but guard anyway
        break;
      } else {
        var constrainedPrompt2 := prompt + generated[..|generated| - |currentConstrainedOut|];
        var next2 := helpers.AdaptiveConstrainedStep(
          lm, parser, constrainedPrompt2, currentConstrainedOut, validTokenGroups, 4.0, 12, eosToken
        );
        steps := steps + 1;
        if next2 == eosToken {
          break;
        } else {
          var isComp3 := parser.IsCompletePrefix(currentConstrainedOut);
          if !isComp3 {
            var ag2, ai2, ac2 := helpers.AppendConstrainedToken(
              lm, parser, generated, currentConstrainedOut, next2
            );
            generated := ag2;
            insideConstrainedOut := ai2;
            currentConstrainedOut := ac2;
          }
        }
      }
    }
  }
}

// Final close attempt if still inside and parser is complete
if insideConstrainedOut && steps < maxSteps {
  var isComp := parser.IsCompletePrefix(currentConstrainedOut);
  if isComp {
    var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + 1;
  }
}

cost := steps;