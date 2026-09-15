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
Capture identity and validation results are documented in the root README.

Each `pypto-lib-operator/golden` directory vendors the pypto-lib golden harness.
Both execution paths require PyPTO and its pinned Simpler runtime; the Simpler
entry uses runtime-directory replay, not PyPTO code generation. No sibling
pypto-lib checkout or machine-specific path is required.

Original source copyright headers are preserved. PyPTO source and generated
artifacts are distributed subject to their upstream CANN Open Software License
Agreement Version 2.0.

## CSA Scheme B

Scheme A is preserved. Scheme B is a local source extension of pypto-lib `092efde7317095c468fba9670efdb3ee6bd9de29`, using the same pinned toolchain. Target: 120 AIC; captured and validated: a2a3, 24 AIC / 48 AIV.

CSA Scheme B uses the mainline canonical start-position set, deduplicated and cycled to 20 requests: [8192,0,2,3,7,127,128,255,511,8192,0,2,3,7,127,128,255,511,8192,0]. KV length is start_pos+2. Generated kernels, ABI metadata and paired captures were refreshed. Capture task: task_20260915_014732_315847915665; validation task: task_20260915_014815_32076747505. Full Scheme B documentation is consolidated into README.md section 3.4.
