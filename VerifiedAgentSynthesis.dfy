module CRANE {
  
  // we alias some types here for readability purposes
  type Token = string
  type Prefix = seq<Token>
  type Logits = map<Token, real>

  // standard sampling strategies that can't be represented easily in dafny due to their probabilistic natures, so we'll need to have them implemented in python
  datatype SamplingStrategy =
    | ArgMax
    | Temperature
    | TopK
    | Nucleus

  // these three functions will all use Lark to determine if the current string is valid/complete under the grammar and what tokens the LLM is allowed to generate, so implementations not necessary (this also means we don't have to represent a grammar in dafny, thank god)
  function {:extern} {:axiom} Parser_ValidPrefix(prefix: Prefix): bool
  function {:extern} {:axiom} Parser_IsComplete(prefix: Prefix): bool
    ensures Parser_IsComplete(prefix) ==> Parser_ValidPrefix(prefix)
  function {:extern} {:axiom} Parser_AllowedNext(prefix: Prefix): set<Token>
    ensures forall t :: t in Parser_AllowedNext(prefix) ==> Parser_ValidPrefix(prefix + [t])
    ensures Parser_ValidPrefix(prefix) ==> (Parser_IsComplete(prefix) || |Parser_AllowedNext(prefix)| > 0)

  // this represents the LLM taking a string and giving logits for each possible next token
  function {:extern} {:axiom} GetLogits(prefix: Prefix): Logits
    ensures forall t :: t is Token ==> t in GetLogits(prefix)

  // checks if every valid next token has a corresponding logit
  predicate ValidLogitsForPrefix(logits: Logits, prefix: Prefix)
  {
    forall t :: t in Parser_AllowedNext(prefix) ==> t in logits
  }

  // this is the part that constrains according to the grammar (checks for all valid possible tokens, their raw scores stay the same, and all others are zeroed out)
  function {:extern} {:axiom} MaskLogits(prefix: Prefix, logits: Logits): Logits
    requires forall t :: t is Token ==> t in logits
    ensures forall t :: t in logits ==> t in MaskLogits(prefix, logits)
    ensures forall t :: t in Parser_AllowedNext(prefix) ==> MaskLogits(prefix, logits)[t] == logits[t]
    ensures forall t :: t !in Parser_AllowedNext(prefix) ==> MaskLogits(prefix, logits)[t] == 0.0

  // the actual sampling function, QWEN should replace the strategy here with one of the above
  function {:extern} {:axiom} SampleWithStrategy(
    logits: Logits,
    candidates: set<Token>,
    strategy: SamplingStrategy
  ): Token
    requires |candidates| > 0
    requires forall t :: t in candidates ==> t in logits
    ensures SampleWithStrategy(logits, candidates, strategy) in candidates

  // the part of the CRANE algorithm that chooses the next token
  function CRANE_ChooseToken(
    prefix: Prefix,
    constrained: bool,
    strategy: SamplingStrategy
  ): Token
    requires Parser_ValidPrefix(prefix)
    requires ValidLogitsForPrefix(GetLogits(prefix), prefix)
    requires |Parser_AllowedNext(prefix)| > 0
    ensures CRANE_ChooseToken(prefix, constrained, strategy) in Parser_AllowedNext(prefix)
  {
    if constrained then
      SampleWithStrategy(
        MaskLogits(prefix, GetLogits(prefix)),
        Parser_AllowedNext(prefix),
        strategy
      )
    else
      SampleWithStrategy(
        GetLogits(prefix),
        Parser_AllowedNext(prefix),
        strategy
      )
  }

  // hacky way to get the boolean for being inside/outside constrained decoding (stateless functions)
  function IsConstrained(currGen: seq<Token>, S1: Token): bool {
    S1 in currGen
  }

  // another hacky way to advance the currGen pointer (stateless functions pt 2)
  function AdvancePointer(
    constrained: bool,
    newTokens: seq<Token>,
    S2: Token,
    pointer: int
  ): int
    requires |newTokens| > 0
  {
    if constrained && newTokens[|newTokens| - 1] == S2 then
      |newTokens|
    else
      pointer
  }

  // the entire CRANE decoding algorithm (slightly modified)
  function CRANE_Decode(
    tokens: Prefix,
    maxSteps: nat,
    S1: Token,
    S2: Token,
    pointer: int,
    strategy: SamplingStrategy
  ): Prefix
    requires Parser_ValidPrefix(tokens)
    requires 0 <= pointer <= |tokens|
    decreases maxSteps
    ensures Parser_ValidPrefix(
      CRANE_Decode(tokens, maxSteps, S1, S2, pointer, strategy)
    )
  {
    if Parser_IsComplete(tokens) then
      tokens
    else if maxSteps == 0 then
      tokens
    else
      var currGen := tokens[pointer..];
      var constrained := IsConstrained(currGen, S1);
      if |Parser_AllowedNext(tokens)| == 0 then
        tokens
      else
        var tok := CRANE_ChooseToken(tokens, constrained, strategy);
        var newTokens := tokens + [tok];
        var pointer := AdvancePointer(constrained, newTokens, S2, pointer);
        if tok == "EOS" then
          newTokens
        else
          CRANE_Decode(newTokens, maxSteps - 1, S1, S2, pointer, strategy)
  }

  // a group of assertions that must hold for the above code
  predicate IsCorrectCRANEDecoder(
    prefix: Prefix,
    result: Prefix,
    maxSteps: nat
  )
  {
    Parser_ValidPrefix(result)
    && |result| >= |prefix| > 0
    && (Parser_IsComplete(result)
        || |result| == |prefix| + maxSteps
        || result[|result| - 1] == "EOS")
    && prefix == result[..|prefix|]
    && (forall k ::
        |prefix| <= k < |result|
        ==> result[k] in Parser_AllowedNext(result[..k]))
  }

  // the induction proof proving the above code follows the specified contracts and is valid
  lemma CRANE_Decode_Correct(
    prefix: Prefix,
    maxSteps: nat,
    S1: Token,
    S2: Token,
    strategy: SamplingStrategy,
    pointer: int := 0
  )
    requires Parser_ValidPrefix(prefix)
    requires ValidLogitsForPrefix(GetLogits(prefix), prefix)
    requires |prefix| > 0
    requires 0 <= pointer <= |prefix|
    decreases maxSteps
    ensures IsCorrectCRANEDecoder(
      prefix,
      CRANE_Decode(prefix, maxSteps, S1, S2, pointer, strategy),
      maxSteps
    )
  {
    if Parser_IsComplete(prefix) {
      assert Parser_ValidPrefix(prefix);
    } else if maxSteps == 0 {
      assert |CRANE_Decode(prefix, 0, S1, S2, pointer, strategy)| == |prefix|;
    } else if |Parser_AllowedNext(prefix)| == 0 {
      assert Parser_IsComplete(prefix);
    } else {
      var currGen := prefix[pointer..];
      var constrained := IsConstrained(currGen, S1);
      var tok := CRANE_ChooseToken(prefix, constrained, strategy);
      assert tok in Parser_AllowedNext(prefix);
      assert Parser_ValidPrefix(prefix + [tok]);
      var newTokens := prefix + [tok];
      var pointer := AdvancePointer(constrained, newTokens, S2, pointer);
      if tok == "EOS" {
        assert Parser_ValidPrefix(prefix + [tok]);
      } else {
        CRANE_Decode_Correct(
          prefix + [tok],
          maxSteps - 1,
          S1,
          S2,
          strategy,
          pointer
        );
      }
    }
  }
}