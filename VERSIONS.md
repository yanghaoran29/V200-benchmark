# Toolchain and source versions

| Component | Revision |
|---|---|
| pypto-lib | `092efde` on `yanghaoran29/pypto-lib-aicore-5x-tiling`, branch `aicore-5x-tiling` |
| PyPTO | `6fdd989fdf45b10011c4bd6b0291a7e6c18fd7e4` |
| Simpler | `39ce891dbb3f665e72e99b4a1387b47012fb18bd` (PyPTO runtime gitlink) |
| PTO ISA | `5a4f74cbf627d4aac2e0ce10d5e0d8b118343265` |
| PTOAS | `v0.61` |
| Archived benchmark | `82f5a5c` (2026-07-14) |

Use the selected PyPTO checkout's installation procedure and pinned dependencies,
with a matching CANN installation. Capture environment: a2a3, 24 AIC / 48 AIV,
CANN 9.0.0. The 120-AIC target has not been measured.

The Qwen capture was produced after `ffb8e36`; its packaged source and C++ are
unchanged by `092efde`, which changed CSA. CSA capture matches `092efde`.
Per-benchmark `PROVENANCE.json` lists upstream source modules, capture identity,
and SHA-256 hashes for packaged source, generated code, metadata and captures.

Each `pypto-lib-operator/golden` directory vendors the pypto-lib golden harness.
Both execution paths require PyPTO and its pinned Simpler runtime; the Simpler
entry uses runtime-directory replay, not PyPTO code generation. No sibling
pypto-lib checkout or machine-specific path is required.

Original source copyright headers are preserved. PyPTO source and generated
artifacts are distributed subject to their upstream CANN Open Software License
Agreement Version 2.0; see the bundled LICENSE files.

## CSA Scheme B

Scheme A is preserved. Scheme B is a local source extension of pypto-lib `092efde7317095c468fba9670efdb3ee6bd9de29`, using the same pinned toolchain. Its source/generated/capture hashes are recorded in `deepseek-v4-csa-b/PROVENANCE.json`. Target: 120 AIC; captured and validated: a2a3, 24 AIC / 48 AIV.

CSA Scheme B default fixture now uses start_pos=8192 for all 20 requests (KV length 8194). The bundled generated kernels and paired capture were refreshed for this input; exact files are identified by the Scheme B provenance manifest.
