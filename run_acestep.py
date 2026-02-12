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
    print("🚀 Starting ACE-Step 1.5 on Intel XPU (Arc A770)...")
    
    # 1. GPU Config
    gpu_config = get_gpu_config()
    print(f"Detected XPU Memory: {gpu_config.gpu_memory_gb:.2f} GB")
    
    # 2. Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--config_path", type=str, default="acestep-v15-turbo")
    parser.add_argument("--lm_model_path", type=str, default="acestep-5Hz-lm-1.7B")
    parser.add_argument("--init_full", action="store_true", help="Initialize all models on startup")
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
            offload_to_cpu=False # 16GB is enough for Turbo + 1.7B LM
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

    # 5. Launch UI
    init_params = {
        'pre_initialized': args.init_full,
        'config_path': args.config_path,
        'device': 'xpu',
        'init_llm': True, # Default to True for 16GB cards
        'lm_model_path': args.lm_model_path,
        'backend': 'pt', # Force PyTorch backend for LLM
        'use_flash_attention': False,
        'offload_to_cpu': False,
        'init_status': init_status,
        'enable_generate': enable_generate,
        'dit_handler': dit_handler,
        'llm_handler': llm_handler,
        'gpu_config': gpu_config,
        'language': 'en'
    }

    print("Launching Gradio Interface...")
    demo = create_gradio_interface(dit_handler, llm_handler, dataset_handler, init_params=init_params)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()