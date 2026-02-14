@echo off
setlocal enabledelayedexpansion
REM ACE-Step Gradio Web UI Launcher - Intel XPU (Arc)
REM Optimized for PyTorch XPU on Intel Arc A770 16GB

REM ==================== Intel XPU Runtime Defaults ====================
if not defined ONEAPI_DEVICE_SELECTOR set ONEAPI_DEVICE_SELECTOR=level_zero:gpu
if not defined SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
if not defined SYCL_CACHE_PERSISTENT set SYCL_CACHE_PERSISTENT=1
if not defined SYCL_PI_LEVEL_ZERO_BATCH_SIZE set SYCL_PI_LEVEL_ZERO_BATCH_SIZE=0
if not defined SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE set SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1
if not defined SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS set SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1
if not defined SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS set SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS=1
if not defined TORCH_XPU_ALLOC_CONF set TORCH_XPU_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
if not defined USE_XETLA set USE_XETLA=OFF
if not defined TORCH_USE_CUDA_DSA set TORCH_USE_CUDA_DSA=1
if not defined PYTORCH_ENABLE_MPS_FALLBACK set PYTORCH_ENABLE_MPS_FALLBACK=1
if not defined OMP_NUM_THREADS set OMP_NUM_THREADS=8
if not defined MKL_NUM_THREADS set MKL_NUM_THREADS=8

REM ==================== Configuration ====================
set PORT=7860
set SERVER_NAME=0.0.0.0
set LANGUAGE=en
REM set SHARE=--share
set CONFIG_PATH=acestep-v15-turbo
set LM_MODEL_PATH=acestep-5Hz-lm-1.7B
REM set OFFLOAD_TO_CPU=true
REM set OFFLOAD_DIT_TO_CPU=true
REM set INIT_LLM=true
set INIT_FULL=
set DOWNLOAD_SOURCE=
set CHECK_UPDATE=true

REM ==================== Startup Update Check ====================
if /i not "%CHECK_UPDATE%"=="true" goto :SkipUpdateCheck
where git >nul 2>&1 || goto :SkipUpdateCheck
cd /d "%~dp0"
git rev-parse --git-dir >nul 2>&1 || goto :SkipUpdateCheck
echo [Update] Checking for updates...
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set UPDATE_BRANCH=%%i
if "!UPDATE_BRANCH!"=="" set UPDATE_BRANCH=main
for /f "tokens=*" %%i in ('git rev-parse --short HEAD 2^>nul') do set UPDATE_LOCAL=%%i
git fetch origin --quiet 2>nul || goto :SkipUpdateCheck
for /f "tokens=*" %%i in ('git rev-parse --short origin/!UPDATE_BRANCH! 2^>nul') do set UPDATE_REMOTE=%%i
if "!UPDATE_REMOTE!"=="" goto :SkipUpdateCheck
if "!UPDATE_LOCAL!"=="!UPDATE_REMOTE!" (
    echo [Update] Already up to date ^(!UPDATE_LOCAL!^).
    echo.
    goto :SkipUpdateCheck
)
echo.
echo ========================================
echo   Update available!
echo ========================================
echo   Current: !UPDATE_LOCAL!  -^>  Latest: !UPDATE_REMOTE!
echo.
git --no-pager log --oneline HEAD..origin/!UPDATE_BRANCH! 2>nul | head -10
echo.
set /p UPDATE_NOW="Update now before starting? (Y/N): "
if /i "!UPDATE_NOW!"=="Y" (
    if exist "%~dp0check_update.bat" (
        call "%~dp0check_update.bat"
    ) else (
        git pull --ff-only origin !UPDATE_BRANCH! 2>nul || echo [Update] Update failed.
    )
) else (
    echo [Update] Skipped. Run check_update.bat to update later.
)
echo.
:SkipUpdateCheck

echo Using currently active environment (e.g., conda). Ensure PyTorch XPU is installed.

python -c "import sys, torch;\nimport textwrap\nmsg='Intel XPU runtime not detected. Please install PyTorch XPU from requirements-xpu.txt'\nassert hasattr(torch,'xpu') and torch.xpu.is_available(), msg\nprops=torch.xpu.get_device_properties(0)\nname=getattr(props,'name','Intel XPU')\nmem=props.total_memory/(1024**3)\nprint(f'[XPU] Detected {name} ({mem:.2f} GB)')" || (
    echo [Error] Intel XPU runtime not detected.
    pause
    exit /b 1
)

echo Starting ACE-Step Gradio Web UI (Intel XPU)...
echo Server will be available at: http://%SERVER_NAME%:%PORT%
echo.
set CMD=python "%~dp0run_acestep.py" --port %PORT% --server-name %SERVER_NAME% --language %LANGUAGE%
if defined SHARE set CMD=!CMD! %SHARE%
if defined CONFIG_PATH set CMD=!CMD! --config_path !CONFIG_PATH!
if defined LM_MODEL_PATH set CMD=!CMD! --lm_model_path !LM_MODEL_PATH!
if defined OFFLOAD_TO_CPU set CMD=!CMD! --offload_to_cpu !OFFLOAD_TO_CPU!
if defined OFFLOAD_DIT_TO_CPU set CMD=!CMD! --offload_dit_to_cpu !OFFLOAD_DIT_TO_CPU!
if defined INIT_LLM set CMD=!CMD! --init_llm !INIT_LLM!
if defined INIT_FULL set CMD=!CMD! --init_full
if defined DOWNLOAD_SOURCE set CMD=!CMD! --download-source !DOWNLOAD_SOURCE!

!CMD!

pause
endlocal
