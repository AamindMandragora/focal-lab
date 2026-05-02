from pathlib import Path
from synthesis.compiler import DafnyCompiler

compiler = DafnyCompiler(dafny_path="dafny", output_dir=Path("outputs/baselines"))
res = compiler.compile_file(Path("dafny/GeneratedCSD.dfy"), "crane_baseline")
print("Success:", res.success)
if not res.success:
    print(res.get_error_summary())
else:
    print("Compiled to:", res.output_dir)
