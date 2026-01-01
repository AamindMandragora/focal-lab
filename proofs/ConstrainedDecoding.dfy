module ConstrainedDecoding {
    // type aliasing, prefix is partial token stream 
    type Token = string
    type Prefix = seq<Token>

    // the requirements that any correct decoder must satisfy
    predicate IsCorrectDecoder(prefix: Prefix, result: Prefix, maxSteps: nat)
    {
        // result is valid under grammar
        Parser_ValidPrefix(result)
        // result is complete under grammar or all steps used up
        && (Parser_IsComplete(result) || |result| == |prefix| + maxSteps)
        // at every generation step the next token must have been a valid next token
        && (forall k :: |prefix| <= k < |result| ==> result[k] in Parser_AllowedNext(result[..k]))
        // result is at least as long as and starts with the prompt
        && |result| >= |prefix| && prefix == result[..|prefix|]
    }

    // return all legal next Tokens under grammar
    function {:extern} {:axiom} Parser_AllowedNext(prefix: Prefix): set<Token>
        // every allowed token combined with the prefix must be valid under grammar
        ensures forall t :: t in Parser_AllowedNext(prefix) ==> Parser_ValidPrefix(prefix + [t])
        // if the prefix is valid, either the prefix is complete or a token can follow it
        ensures Parser_ValidPrefix(prefix) ==> (Parser_IsComplete(prefix) || |Parser_AllowedNext(prefix)| > 0)

    // checks if prefix is valid under grammar
    function {:extern} Parser_ValidPrefix(prefix: Prefix): bool

    // checks if prefix is complete under grammar
    function {:extern} {:axiom} Parser_IsComplete(prefix: Prefix): bool
        // complete prefixes must be valid under grammar
        ensures Parser_ValidPrefix(prefix)

    // returns LLM suggested token to follow prefix based on some arbitrary algorithm
    function {:extern} {:axiom} Generator_ChooseToken(prefix: Prefix, allowed_tokens: set<Token>): Token
        // must have a token to choose from
        requires |allowed_tokens| > 0
        // ensures chosen token was in the allowed tokens
        ensures Generator_ChooseToken(prefix, allowed_tokens) in allowed_tokens
    
    // models recursive constrained decoder
    function ConstrainedDecode(prefix: Prefix, maxSteps: nat): (result: Prefix)
        // cannot have negative steps
        requires maxSteps >= 0
        // prompt must be valid in the grammar
        requires Parser_ValidPrefix(prefix)
        // each recursive call a step is consumed
        decreases maxSteps
        // result must be valid under grammar
        ensures Parser_ValidPrefix(result)
        // either result is complete in the grammar or all steps used
        ensures Parser_IsComplete(result) || |result| == |prefix| + maxSteps
    {
    if Parser_IsComplete(prefix) then
        // complete sentence generated, returns
        prefix
    else if maxSteps == 0 then
        // no more steps allowed, returns whatever was generated
        prefix
    else
        // finds allowed next tokens
        var allowed_tokens := Parser_AllowedNext(prefix);
        if |allowed_tokens| == 0 then 
            // no tokens allowed, returns whatever was generated
            prefix
        else
            // generate a token using strategy
            var generated_token := Generator_ChooseToken(prefix, allowed_tokens);
            // consume a step and recurse on the prompt joined with the generated token
            ConstrainedDecode(prefix + [generated_token], maxSteps - 1)
    }

    // verification of ConstrainedDecode
    lemma ConstrainedDecode_Correct(prefix: Prefix, maxSteps: nat)
        // prompt must be valid under grammar
        requires Parser_ValidPrefix(prefix)
        // induction on maxSteps
        decreases maxSteps
        // follows predicate at the top
        ensures IsCorrectDecoder(prefix, ConstrainedDecode(prefix, maxSteps), maxSteps)
    {
        if Parser_IsComplete(prefix) {
            // complete prompts must be valid
            assert Parser_ValidPrefix(prefix);
            // prompt is a valid generation of itself
            assert IsCorrectDecoder(prefix, prefix, maxSteps);
        } else if maxSteps == 0 {
            // prompt must be valid
            assert Parser_ValidPrefix(prefix);
            // no tokens should be generated as there are no remaining steps
            assert |ConstrainedDecode(prefix, maxSteps)| == |prefix|;
            // no generation must satisfy predicate
            assert IsCorrectDecoder(prefix, prefix, 0);
        } else {
            // get tokens allowed to follow prompt
            var allowed_tokens := Parser_AllowedNext(prefix);
            if |allowed_tokens| == 0 {
                // no tokens can follow prompt, so must be complete
                assert Parser_IsComplete(prefix);
                // prompt is a valid generation of itself
                assert IsCorrectDecoder(prefix, prefix, 0);
            } else {
                // generate token from allowed
                var generated_token := Generator_ChooseToken(prefix, allowed_tokens);
                // generated token must be in allowed
                assert generated_token in allowed_tokens;
                // prefix + token must be valid in the grammar
                assert Parser_ValidPrefix(prefix + [generated_token]);
                // get the rest of the generation
                var subresult := ConstrainedDecode(prefix + [generated_token], maxSteps - 1);
                // subresult must be at least one token longer than the prompt (generated_token)
                assert |subresult| >= |prefix| + 1;
                // the first |prompt| tokens in subresult must equal prompt
                assert prefix == subresult[..|prefix|];
                // the |prompt| token of the subresult must be the generated token
                assert subresult[|prefix|] == generated_token;
                // the generated token must be in the allowed next tokens of prompt
                assert generated_token in Parser_AllowedNext(prefix);
                // equivalent to above but replaced prompt with the first |prompt| tokens of subresult
                assert generated_token in Parser_AllowedNext(subresult[..|prefix|]);
                // each token following after prompt in subresult must be in the allowed next tokens of the preceding prefix
                assert (forall k :: |prefix| <= k < |subresult| ==> subresult[k] in Parser_AllowedNext(subresult[..k]));
                // inductive step
                ConstrainedDecode_Correct(prefix + [generated_token], maxSteps - 1);
            }
        }
    }

    // ---------------------------------------------------------------------
    // CSD-parameterized constrained decoding (window-local)
    //
    // This variant models a constrained decoder whose "allowed next tokens" and
    // "choice of token" are provided by an external, compositional CSD policy.
    // ---------------------------------------------------------------------

    // Abstract handle for a constrained-decoding strategy component.
    // (In the whole-loop model in `proofs/CSD.dfy`, this corresponds to the
    // constrained fallback strategy used after retries.)
    type Strategy = int
    type StrategyState = int

    // CSD-provided allowed-next set for a constrained-window prefix.
    //
    // Contracts are chosen so that we can reuse IsCorrectDecoder:
    // - any allowed token must be a grammar-allowed next token, and
    // - if a prefix is valid and incomplete, the CSD must allow progress.
    function {:extern} {:axiom} Strategy_AllowedNext(s: Strategy, st: StrategyState, prefix: Prefix): set<Token>
        ensures forall t :: t in Strategy_AllowedNext(s, st, prefix) ==> t in Parser_AllowedNext(prefix)
        ensures Parser_ValidPrefix(prefix) ==> (Parser_IsComplete(prefix) || |Strategy_AllowedNext(s, st, prefix)| > 0)

    function {:extern} {:axiom} Strategy_ChooseToken(s: Strategy, st: StrategyState, prefix: Prefix): Token
        requires |Strategy_AllowedNext(s, st, prefix)| > 0
        ensures Strategy_ChooseToken(s, st, prefix) in Strategy_AllowedNext(s, st, prefix)

    function Strategy_ConstrainedDecode(s: Strategy, st: StrategyState, prefix: Prefix, maxSteps: nat): (result: Prefix)
        requires maxSteps >= 0
        requires Parser_ValidPrefix(prefix)
        decreases maxSteps
        ensures Parser_ValidPrefix(result)
        ensures Parser_IsComplete(result) || |result| == |prefix| + maxSteps
    {
        if Parser_IsComplete(prefix) then
            prefix
        else if maxSteps == 0 then
            prefix
        else
            var allowed_tokens := Strategy_AllowedNext(s, st, prefix);
            if |allowed_tokens| == 0 then
                // By the CSD_AllowedNext contract, this implies the prefix is complete.
                prefix
            else
                var generated_token := Strategy_ChooseToken(s, st, prefix);
                Strategy_ConstrainedDecode(s, st, prefix + [generated_token], maxSteps - 1)
    }

    lemma Strategy_ConstrainedDecode_Correct(s: Strategy, st: StrategyState, prefix: Prefix, maxSteps: nat)
        requires Parser_ValidPrefix(prefix)
        decreases maxSteps
        ensures IsCorrectDecoder(prefix, Strategy_ConstrainedDecode(s, st, prefix, maxSteps), maxSteps)
    {
        if Parser_IsComplete(prefix) {
            assert Parser_ValidPrefix(prefix);
            assert IsCorrectDecoder(prefix, prefix, maxSteps);
        } else if maxSteps == 0 {
            assert Parser_ValidPrefix(prefix);
            assert |Strategy_ConstrainedDecode(s, st, prefix, maxSteps)| == |prefix|;
            assert IsCorrectDecoder(prefix, prefix, 0);
        } else {
            var allowed_tokens := Strategy_AllowedNext(s, st, prefix);
            if |allowed_tokens| == 0 {
                // From Parser_ValidPrefix(prefix) and the CSD_AllowedNext contract:
                assert Parser_IsComplete(prefix);
                assert IsCorrectDecoder(prefix, prefix, 0);
            } else {
                var generated_token := Strategy_ChooseToken(s, st, prefix);
                assert generated_token in allowed_tokens;
                assert generated_token in Parser_AllowedNext(prefix);
                assert Parser_ValidPrefix(prefix + [generated_token]);
                var subresult := Strategy_ConstrainedDecode(s, st, prefix + [generated_token], maxSteps - 1);
                assert |subresult| >= |prefix| + 1;
                assert prefix == subresult[..|prefix|];
                assert subresult[|prefix|] == generated_token;
                assert generated_token in Parser_AllowedNext(subresult[..|prefix|]);
                assert (forall k :: |prefix| <= k < |subresult| ==> subresult[k] in Parser_AllowedNext(subresult[..k]));
                Strategy_ConstrainedDecode_Correct(s, st, prefix + [generated_token], maxSteps - 1);
            }
        }
    }
}