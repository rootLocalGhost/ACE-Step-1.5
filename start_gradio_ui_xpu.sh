#!/usr/bin/env bash
# ACE-Step Gradio Web UI Launcher - Intel XPU (Arc)
# Optimized for PyTorch XPU on Intel Arc A770 16GB

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== Intel XPU Runtime Defaults ====================
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-"level_zero:gpu"}
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=${SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS:-"1"}
export SYCL_CACHE_PERSISTENT=${SYCL_CACHE_PERSISTENT:-"1"}
export SYCL_PI_LEVEL_ZERO_BATCH_SIZE=${SYCL_PI_LEVEL_ZERO_BATCH_SIZE:-"0"}
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=${SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE:-"1"}
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=${SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS:-"1"}
export SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS=${SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS:-"1"}
export TORCH_XPU_ALLOC_CONF=${TORCH_XPU_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb:64"}
export USE_XETLA=${USE_XETLA:-"OFF"}
export TORCH_USE_CUDA_DSA=${TORCH_USE_CUDA_DSA:-"1"}
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-"1"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"8"}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-"8"}

# ==================== Configuration ====================
PORT=7860
SERVER_NAME="0.0.0.0"
LANGUAGE="en"
SHARE=""
CONFIG_PATH="acestep-v15-turbo"
LM_MODEL_PATH="acestep-5Hz-lm-1.7B"
OFFLOAD_TO_CPU=""
OFFLOAD_DIT_TO_CPU=""
INIT_LLM=""
INIT_FULL=""
CHECK_UPDATE="true"
DOWNLOAD_SOURCE=""

# ==================== Helpers ====================
_verify_env() {
    # Use the currently activated environment (e.g., conda). Expect torch with XPU support installed.
    command -v python &>/dev/null || { echo "[Error] python not found in PATH. Activate your conda env first."; exit 1; }
    python - <<'PY'
import sys
try:
    import torch
except Exception as exc:  # noqa: BLE001
    sys.exit(f"PyTorch not available in current environment: {exc}")

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    sys.exit("Intel XPU runtime not detected. Activate your conda env with PyTorch XPU installed.")

props = torch.xpu.get_device_properties(0)
name = getattr(props, "name", "Intel XPU")
mem = props.total_memory / (1024**3)
print(f"[XPU] Detected {name} ({mem:.2f} GB)")
PY
}

_startup_update_check() {
    [[ "$CHECK_UPDATE" != "true" ]] && return 0
    command -v git &>/dev/null || return 0
    cd "$SCRIPT_DIR" || return 0
    git rev-parse --git-dir &>/dev/null || return 0
    local branch commit remote_commit
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")"
    commit="$(git rev-parse --short HEAD 2>/dev/null || echo "")"
    [[ -z "$commit" ]] && return 0
    echo "[Update] Checking for updates..."
    git fetch origin --quiet 2>/dev/null || { echo "[Update] Network unreachable, skipping."; echo; return 0; }
    remote_commit="$(git rev-parse --short "origin/$branch" 2>/dev/null || echo "")"
    if [[ -z "$remote_commit" || "$commit" == "$remote_commit" ]]; then
        echo "[Update] Already up to date ($commit)."; echo; return 0
    fi
    echo
    echo "========================================"
    echo "  Update available!"
    echo "========================================"
    echo "  Current: $commit  ->  Latest: $remote_commit"
    echo
    git --no-pager log --oneline "HEAD..origin/$branch" 2>/dev/null | head -10
    echo
    read -rp "Update now before starting? (Y/N): " update_choice
    if [[ "${update_choice^^}" == "Y" ]]; then
        if [[ -f "$SCRIPT_DIR/check_update.sh" ]]; then
            bash "$SCRIPT_DIR/check_update.sh"
        else
            git pull --ff-only origin "$branch" || echo "[Update] Update failed."
        fi
    else
        echo "[Update] Skipped. Run ./check_update.sh to update later."
    fi
    echo
}

# ==================== Main ====================
_startup_update_check

_verify_env

echo "Starting ACE-Step Gradio Web UI (Intel XPU)..."
echo "Server: http://${SERVER_NAME}:${PORT}"
echo

CMD=(python "$SCRIPT_DIR/run_acestep.py" --port "$PORT" --server-name "$SERVER_NAME" --language "$LANGUAGE")
[[ -n "$SHARE" ]] && CMD+=("$SHARE")
[[ -n "$CONFIG_PATH" ]] && CMD+=("--config_path" "$CONFIG_PATH")
[[ -n "$LM_MODEL_PATH" ]] && CMD+=("--lm_model_path" "$LM_MODEL_PATH")
[[ -n "$OFFLOAD_TO_CPU" ]] && CMD+=("--offload_to_cpu" "$OFFLOAD_TO_CPU")
[[ -n "$OFFLOAD_DIT_TO_CPU" ]] && CMD+=("--offload_dit_to_cpu" "$OFFLOAD_DIT_TO_CPU")
[[ -n "$INIT_LLM" ]] && CMD+=("--init_llm" "$INIT_LLM")
[[ -n "$INIT_FULL" ]] && CMD+=("--init_full")
[[ -n "$DOWNLOAD_SOURCE" ]] && CMD+=("--download-source" "$DOWNLOAD_SOURCE")

cd "$SCRIPT_DIR" && "${CMD[@]}"
