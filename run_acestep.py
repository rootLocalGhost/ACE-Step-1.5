import os
import sys
import argparse
import warnings
import logging

# --- WARNING SUPPRESSION ---
# Filter out "CUDA is not available" warnings from diffusers
warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA is not available.*")
# Filter out "torch_dtype is deprecated" warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch_dtype is deprecated.*")
# Suppress specific library logging
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
# ---------------------------

XPU_ENV_DEFAULTS = {
    # Prefer Level Zero GPU backend and avoid CPU fallback
    "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
    # Reduce launch overhead and improve latency for real-time generation
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
    # Keep compiled kernels cached across runs when possible
    "SYCL_CACHE_PERSISTENT": "1",  # Enable persistent cache to reuse compiled kernels across runs
    "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE": "1",  # Use copy engines and device-scope events to reduce stalls
    "SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS": "1",
    "SYCL_PI_LEVEL_ZERO_BATCH_SIZE": "0",  # Allow driver to select best batch size (0 = auto)
    "SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS": "1",  # Relax allocation limits to reduce fragmentation issues
    # Make allocator friendlier to long-running audio jobs
    "TORCH_XPU_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:64",
    # Intel Arc GPUs do not benefit from XeTLA in this workload
    "USE_XETLA": "OFF",
    "TORCH_USE_CUDA_DSA": "1",  # Native PyTorch opts
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "OMP_NUM_THREADS": "8",  # Threading defaults (tunable per host)
    "MKL_NUM_THREADS": "8",
}


def configure_xpu_environment() -> None:
    """Apply conservative Intel XPU runtime defaults before importing torch."""
    for key, value in XPU_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


configure_xpu_environment()

import torch

# Ensure we can import modules from the current directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our custom XPU handlers
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.dataset_handler import DatasetHandler
from acestep.gradio_ui import create_gradio_interface
from acestep.gpu_config import get_gpu_config

def main():
    """Launch ACE-Step Gradio UI optimized for Intel XPU (Arc A770 class)."""
    print("🚀 Starting ACE-Step 1.5 on Intel XPU (Arc A770)...")
    
    # 1. GPU Config
    gpu_config = get_gpu_config()
    print(f"Detected XPU Memory: {gpu_config.gpu_memory_gb:.2f} GB (tier={gpu_config.tier})")
    print(f"Default offload_to_cpu: {gpu_config.offload_to_cpu_default}")
    torch.set_float32_matmul_precision("medium")
    
    # 2. Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh", "he", "ja"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--config_path", type=str, default="acestep-v15-turbo")
    parser.add_argument("--lm_model_path", type=str, default="acestep-5Hz-lm-1.7B")
    parser.add_argument("--init_full", action="store_true", help="Initialize all models on startup")
    parser.add_argument("--init_llm", type=lambda x: x.lower() in ["true", "1", "yes"], default=None,
                        help="Initialize LLM on startup (default: auto based on GPU tier)")
    parser.add_argument("--offload_to_cpu", type=lambda x: x.lower() in ["true", "1", "yes"],
                        default=gpu_config.offload_to_cpu_default,
                        help="Offload models to CPU when idle (default: tier-based)")
    parser.add_argument("--offload_dit_to_cpu", type=lambda x: x.lower() in ["true", "1", "yes"],
                        default=gpu_config.offload_dit_to_cpu_default,
                        help="Offload DiT to CPU when idle (effective only with --offload_to_cpu)")
    args = parser.parse_args()

    # 3. Initialize Handlers
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()
    dataset_handler = DatasetHandler()
    
    init_status = ""
    enable_generate = False
    
    # 4. Optional Immediate Initialization
    if args.init_full:
        print("Initializing models immediately...")
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize DiT (Diffusion Transformer)
        status, success = dit_handler.initialize_service(
            project_root=project_root,
            config_path=args.config_path,
            device="xpu",
            use_flash_attention=False, # Must be false for XPU
            offload_to_cpu=args.offload_to_cpu,
            offload_dit_to_cpu=args.offload_dit_to_cpu
        )
        init_status += status
        
        # Initialize LLM (PyTorch backend forced by our XPU class)
        print("Initializing LLM...")
        lm_status, lm_success = llm_handler.initialize(
            checkpoint_dir=os.path.join(project_root, "checkpoints"),
            lm_model_path=args.lm_model_path,
            device="xpu",
            dtype=torch.bfloat16
        )
        init_status += f"\n{lm_status}"
        enable_generate = success and lm_success
    elif args.init_llm is None:
        # Align default with GPU tier when not pre-initializing
        args.init_llm = gpu_config.init_lm_default

    if args.init_llm is None:
        args.init_llm = gpu_config.init_lm_default

    # 5. Launch UI
    init_params = {
        'pre_initialized': args.init_full,
        'config_path': args.config_path,
        'device': 'xpu',
        'init_llm': args.init_llm,
        'lm_model_path': args.lm_model_path,
        'backend': 'pt', # Force PyTorch backend for LLM
        'use_flash_attention': False,
        'offload_to_cpu': args.offload_to_cpu,
        'offload_dit_to_cpu': args.offload_dit_to_cpu,
        'init_status': init_status,
        'enable_generate': enable_generate,
        'dit_handler': dit_handler,
        'llm_handler': llm_handler,
        'gpu_config': gpu_config,
        'language': args.language
    }

    print("Launching Gradio Interface...")
    demo = create_gradio_interface(dit_handler, llm_handler, dataset_handler, init_params=init_params, language=args.language)
    demo.launch(server_name=args.server_name, server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()