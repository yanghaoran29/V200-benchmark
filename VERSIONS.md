# Toolchain versions

Pinned at 2026-09-14 for the CSA / Qwen3 `_decode_layer` → V200 simpler sample work.

| Component | Version | Source |
|---|---|---|
| pypto-lib | `3ac8fd8815402a0dc17914dc8c9440d9de5b7329` (`3ac8fd8`, main) | this checkout |
| pypto | `6fdd989fdf45b10011c4bd6b0291a7e6c18fd7e4` (`6fdd989`, main) | origin/main at pin time |
| simpler | `39ce891dbb3f665e72e99b4a1387b47012fb18bd` (`39ce891`, detached) | pypto `runtime/` submodule |
| pto-isa | `5a4f74cbf627d4aac2e0ce10d5e0d8b118343265` (`5a4f74cb`, detached) | pypto `runtime/pto_isa.pin` |
| ptoas | `v0.61` (`ptoas 0.61`) | pypto `toolchain/versions.env` |

## Local roots

- `PYPTO_ROOT=/data/y00955915/Desktop/pypto-lib-09141/pypto`
- `PTOAS_ROOT=/data/y00955915/Desktop/pypto-lib-09141/ptoas-bin`
- `PTO_ISA_ROOT=/data/y00955915/Desktop/pypto-lib-09141/.venv/lib/python3.11/site-packages/simpler_setup/_assets/build/pto-isa`
- activate: `source /data/y00955915/Desktop/pypto-lib-09141/activate.sh`
- GitHub: `source ~/.config/pto-auth-proxy/env.sh` (port 20809)
