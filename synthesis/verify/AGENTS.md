# AGENTS.md — `synthesis/verify/`

## Scope

**Dafny verification** and **Dafny-to-Python compilation** wrappers around the verified strategy template.

## Rules

- Follow root **`AGENTS.md`**: do not alter formal contracts in **`library/`** unless necessary for a verified, agreed change.
- Verification failures are first-class inputs to refinement; keep error capture and reporting stable for **`generate/`** to consume.
- Compilation output must remain compatible with **`evaluate/`** runtime imports.

## See also

- **`README.md`** in this folder for subprocess and path behavior.
- **`library/README.md`** and **`library/AGENTS.md`** for the Dafny template surface.
