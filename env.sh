#!/bin/bash
# ROCmFPX (gfx1100) run environment for this machine.
# Usage: source ~/ROCmFPX/env.sh   (then run llama-server/llama-cli from ~/ROCmFPX/build-rdna3/bin)
# NOTE: never leave the stale llama-rocm/llama-b10064 dir in LD_LIBRARY_PATH - it
# shadows this build's libs and causes undefined-symbol errors.
export ROCMFPX_BIN=/home/sopdet/ROCmFPX/build-rdna3/bin
export LD_LIBRARY_PATH="/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib/rocm_sysdeps/lib:/opt/rocm/core-7.14/lib/llvm/lib:/opt/rocm/core-7.14/lib/llvm/lib/clang/23/lib/linux:/home/sopdet/.local/rocm-staging/lib"
export PATH="$ROCMFPX_BIN:$PATH"
