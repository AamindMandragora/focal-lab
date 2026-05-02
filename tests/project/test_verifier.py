from verification.verifier import DafnyVerifier, VerificationError, VerificationResult


def test_verification_summary_mentions_python_line():
    result = VerificationResult(
        success=False,
        errors=[
            VerificationError(
                file="GeneratedCSD.dfy",
                line=42,
                column=7,
                message="some verifier issue",
                python_line=13,
            )
        ],
    )

    summary = result.get_error_summary()

    assert "Dafny line 42, Column 7; Python line 13" in summary


def test_verifier_maps_tagged_dafny_lines_back_to_python_lines():
    verifier = DafnyVerifier.__new__(DafnyVerifier)
    dafny_source = """method Demo()
{
  // Python line 10
  var x := 0;
  // Python line 14
  assert x == 0;
}
"""

    line_map = verifier._build_python_line_map(dafny_source)
    errors = [
        VerificationError(
            file="GeneratedCSD.dfy",
            line=6,
            column=3,
            message="assertion might not hold",
        )
    ]

    verifier._attach_python_line_numbers(errors, line_map)

    assert errors[0].python_line == 14
