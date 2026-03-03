"""FastAPI server for ACE-Step V1.5.

Endpoints:
- POST /release_task          Create music generation task
- POST /query_result          Batch query task results
- POST /create_random_sample  Generate random music parameters via LLM
- POST /format_input          Format and enhance lyrics/caption via LLM
- GET  /v1/models             List available models
- GET  /v1/audio              Download audio file
- GET  /health                Health check

NOTE:
- In-memory queue and job store -> run uvicorn with workers=1.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
import time
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import torch
from loguru import logger

try:
    from dotenv import load_dotenv
except ImportError:  # Optional dependency
    load_dotenv = None  # type: ignore

from fastapi import FastAPI
from acestep.api.train_api_service import (
    initialize_training_state,
)
from acestep.api.jobs.store import _JobStore
from acestep.api.log_capture import install_log_capture
from acestep.api.route_setup import configure_api_routes
from acestep.api.server_cli import run_api_server_main
from acestep.api.lifespan_runtime import initialize_lifespan_runtime
from acestep.api.job_generation_setup import build_generation_setup
from acestep.api.job_model_selection import select_generation_handler
from acestep.api.job_llm_preparation import (
    ensure_llm_ready_for_request as _ensure_llm_ready_for_request,
    prepare_llm_generation_inputs as _prepare_llm_generation_inputs,
)
from acestep.api.job_result_payload import build_generation_success_response
from acestep.api.job_runtime_state import (
    cleanup_job_temp_files as _cleanup_job_temp_files_state,
    ensure_models_initialized as _ensure_models_initialized,
    update_progress_job_cache as _update_progress_job_cache,
    update_terminal_job_cache as _update_terminal_job_cache,
)
from acestep.api.startup_model_init import initialize_models_at_startup
from acestep.api.worker_runtime import start_worker_tasks, stop_worker_tasks
from acestep.api.server_utils import (
    env_bool as _env_bool,
    get_model_name as _get_model_name,
    is_instrumental as _is_instrumental,
    map_status as _map_status,
    parse_description_hints as _parse_description_hints,
    parse_timesteps as _parse_timesteps,
)
from acestep.api.http.auth import (
    set_api_key,
    verify_api_key,
    verify_token_from_request,
)
from acestep.api.http.release_task_audio_paths import (
    save_upload_to_temp as _save_upload_to_temp,
    validate_audio_path as _validate_audio_path,
)
from acestep.api.http.release_task_models import GenerateMusicRequest
from acestep.api.http.release_task_param_parser import (
    RequestParser,
    _to_float as _request_to_float,
    _to_int as _request_to_int,
)
from acestep.api.runtime_helpers import (
    append_jsonl as _runtime_append_jsonl,
    atomic_write_json as _runtime_atomic_write_json,
    start_tensorboard as _runtime_start_tensorboard,
    stop_tensorboard as _runtime_stop_tensorboard,
    temporary_llm_model as _runtime_temporary_llm_model,
)
from acestep.api.model_download import (
    ensure_model_downloaded as _ensure_model_downloaded,
)

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.constants import (
    DEFAULT_DIT_INSTRUCTION,
    TASK_INSTRUCTIONS,
)
from acestep.inference import (
    generate_music,
    create_sample,
    format_sample,
)
from acestep.ui.gradio.events.results_handlers import _build_generation_info

def _get_project_root() -> str:
    current_file = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(current_file))


# =============================================================================
# Constants
# =============================================================================

RESULT_KEY_PREFIX = "ace_step_v1.5_"
RESULT_EXPIRE_SECONDS = 7 * 24 * 60 * 60  # 7 days
TASK_TIMEOUT_SECONDS = 3600  # 1 hour
JOB_STORE_CLEANUP_INTERVAL = 300  # 5 minutes - interval for cleaning up old jobs
JOB_STORE_MAX_AGE_SECONDS = 86400  # 24 hours - completed jobs older than this will be cleaned

LM_DEFAULT_TEMPERATURE = 0.85
LM_DEFAULT_CFG_SCALE = 2.5
LM_DEFAULT_TOP_P = 0.9


def _wrap_response(data: Any, code: int = 200, error: Optional[str] = None) -> Dict[str, Any]:
    """Wrap response data in standard format."""
    return {
        "data": data,
        "code": code,
        "error": error,
        "timestamp": int(time.time() * 1000),
        "extra": None,
    }


# =============================================================================
# Example Data for Random Sample
# =============================================================================

SIMPLE_MODE_EXAMPLES_DIR = os.path.join(_get_project_root(), "examples", "simple_mode")
CUSTOM_MODE_EXAMPLES_DIR = os.path.join(_get_project_root(), "examples", "text2music")


def _load_all_examples(sample_mode: str = "simple_mode") -> List[Dict[str, Any]]:
    """Load all example data files from the examples directory."""
    examples = []
    examples_dir = SIMPLE_MODE_EXAMPLES_DIR if sample_mode == "simple_mode" else CUSTOM_MODE_EXAMPLES_DIR
    pattern = os.path.join(examples_dir, "example_*.json")

    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                examples.append(data)
        except Exception as e:
            print(f"[API Server] Failed to load example file {filepath}: {e}")

    return examples


# Pre-load example data at module load time
SIMPLE_EXAMPLE_DATA: List[Dict[str, Any]] = _load_all_examples(sample_mode="simple_mode")
CUSTOM_EXAMPLE_DATA: List[Dict[str, Any]] = _load_all_examples(sample_mode="custom_mode")


_project_env_loaded = False


def _load_project_env() -> None:
    """Load .env at most once per process to avoid epoch-boundary stalls (e.g. Windows LoRA training)."""
    global _project_env_loaded
    if _project_env_loaded or load_dotenv is None:
        return
    try:
        project_root = _get_project_root()
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
        _project_env_loaded = True
    except Exception:
        # Optional best-effort: continue even if .env loading fails.
        pass


_load_project_env()


log_buffer, _stderr_proxy = install_log_capture(logger, sys.stderr)
sys.stderr = _stderr_proxy


def create_app() -> FastAPI:
    store = _JobStore()

    # API Key authentication (from environment variable)
    api_key = os.getenv("ACESTEP_API_KEY", None)
    set_api_key(api_key)

    QUEUE_MAXSIZE = int(os.getenv("ACESTEP_QUEUE_MAXSIZE", "200"))
    WORKER_COUNT = int(os.getenv("ACESTEP_QUEUE_WORKERS", "1"))  # Single GPU recommended

    INITIAL_AVG_JOB_SECONDS = float(os.getenv("ACESTEP_AVG_JOB_SECONDS", "5.0"))
    AVG_WINDOW = int(os.getenv("ACESTEP_AVG_WINDOW", "50"))

    def _path_to_audio_url(path: str) -> str:
        """Convert local file path to downloadable relative URL"""
        if not path:
            return path
        if path.startswith("http://") or path.startswith("https://"):
            return path
        encoded_path = urllib.parse.quote(path, safe="")
        return f"/v1/audio?path={encoded_path}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = initialize_lifespan_runtime(
            app=app,
            store=store,
            queue_maxsize=QUEUE_MAXSIZE,
            avg_window=AVG_WINDOW,
            initial_avg_job_seconds=INITIAL_AVG_JOB_SECONDS,
            get_project_root=_get_project_root,
            initialize_training_state_fn=initialize_training_state,
            ace_handler_cls=AceStepHandler,
            llm_handler_cls=LLMHandler,
        )
        handler = runtime.handler
        llm_handler = runtime.llm_handler
        handler2 = runtime.handler2
        handler3 = runtime.handler3
        config_path2 = runtime.config_path2
        config_path3 = runtime.config_path3
        executor = runtime.executor

        async def _run_one_job(job_id: str, req: GenerateMusicRequest) -> None:
            job_store: _JobStore = app.state.job_store
            llm: LLMHandler = app.state.llm_handler
            executor: ThreadPoolExecutor = app.state.executor

            await _ensure_models_initialized(app.state)
            job_store.mark_running(job_id)
            _update_progress_job_cache(
                app_state=app.state,
                store=store,
                job_id=job_id,
                progress=0.01,
                stage="running",
                map_status=_map_status,
                result_key_prefix=RESULT_KEY_PREFIX,
                result_expire_seconds=RESULT_EXPIRE_SECONDS,
            )

            selected_handler, selected_model_name = select_generation_handler(
                app_state=app.state,
                requested_model=req.model,
                get_model_name=_get_model_name,
                job_id=job_id,
                log_fn=print,
            )

            # Use selected handler for generation
            h: AceStepHandler = selected_handler

            def _blocking_generate() -> Dict[str, Any]:
                """Generate music using unified inference logic from acestep.inference"""

                def _ensure_llm_ready() -> None:
                    _ensure_llm_ready_for_request(
                        app_state=app.state,
                        llm_handler=llm,
                        req=req,
                        get_project_root=_get_project_root,
                        get_model_name=_get_model_name,
                        ensure_model_downloaded=_ensure_model_downloaded,
                        env_bool=_env_bool,
                        log_fn=print,
                    )

                prepared_inputs = _prepare_llm_generation_inputs(
                    app_state=app.state,
                    llm_handler=llm,
                    req=req,
                    selected_handler_device=h.device,
                    parse_description_hints=_parse_description_hints,
                    create_sample_fn=create_sample,
                    format_sample_fn=format_sample,
                    ensure_llm_ready_fn=_ensure_llm_ready,
                    log_fn=print,
                )

                lm_top_k = prepared_inputs.lm_top_k
                lm_top_p = prepared_inputs.lm_top_p
                thinking = prepared_inputs.thinking
                sample_mode = prepared_inputs.sample_mode
                use_cot_caption = prepared_inputs.use_cot_caption
                use_cot_language = prepared_inputs.use_cot_language
                caption = prepared_inputs.caption
                lyrics = prepared_inputs.lyrics
                bpm = prepared_inputs.bpm
                key_scale = prepared_inputs.key_scale
                time_signature = prepared_inputs.time_signature
                audio_duration = prepared_inputs.audio_duration
                original_prompt = prepared_inputs.original_prompt
                original_lyrics = prepared_inputs.original_lyrics
                format_has_duration = prepared_inputs.format_has_duration

                generation_setup = build_generation_setup(
                    req=req,
                    caption=caption,
                    lyrics=lyrics,
                    bpm=bpm,
                    key_scale=key_scale,
                    time_signature=time_signature,
                    audio_duration=audio_duration,
                    thinking=thinking,
                    sample_mode=sample_mode,
                    format_has_duration=format_has_duration,
                    use_cot_caption=use_cot_caption,
                    use_cot_language=use_cot_language,
                    lm_top_k=lm_top_k,
                    lm_top_p=lm_top_p,
                    parse_timesteps=_parse_timesteps,
                    is_instrumental=_is_instrumental,
                    default_dit_instruction=DEFAULT_DIT_INSTRUCTION,
                    task_instructions=TASK_INSTRUCTIONS,
                )
                params = generation_setup.params
                config = generation_setup.config

                # Check LLM initialization status
                llm_is_initialized = getattr(app.state, "_llm_initialized", False)
                llm_to_pass = llm if llm_is_initialized else None

                # Progress callback for API polling
                last_progress = {"value": -1.0, "time": 0.0, "stage": ""}

                def _progress_cb(value: float, desc: str = "") -> None:
                    now = time.time()
                    try:
                        value_f = max(0.0, min(1.0, float(value)))
                    except Exception:
                        value_f = 0.0
                    stage = desc or last_progress["stage"] or "running"
                    # Throttle updates to avoid excessive cache writes
                    if (
                        value_f - last_progress["value"] >= 0.01
                        or stage != last_progress["stage"]
                        or (now - last_progress["time"]) >= 0.5
                    ):
                        last_progress["value"] = value_f
                        last_progress["time"] = now
                        last_progress["stage"] = stage
                        job_store.update_progress(job_id, value_f, stage=stage)
                        _update_progress_job_cache(
                            app_state=app.state,
                            store=store,
                            job_id=job_id,
                            progress=value_f,
                            stage=stage,
                            map_status=_map_status,
                            result_key_prefix=RESULT_KEY_PREFIX,
                            result_expire_seconds=RESULT_EXPIRE_SECONDS,
                        )

                if req.full_analysis_only:
                    store.update_progress_text(job_id, "Starting Deep Analysis...")
                    # Step A: Convert source audio to semantic codes
                    # We use params.src_audio which is the server-side path
                    audio_codes = h.convert_src_audio_to_codes(params.src_audio)

                    if not audio_codes or audio_codes.startswith("❌"):
                        raise RuntimeError(f"Audio encoding failed: {audio_codes}")

                    # Step B: LLM Understanding of those specific codes
                    # This yields the deep metadata and lyrics transcription
                    metadata_dict, status_string = llm_to_pass.understand_audio_from_codes(
                        audio_codes=audio_codes,
                        temperature=0.3,
                        use_constrained_decoding=True,
                        constrained_decoding_debug=config.constrained_decoding_debug
                    )

                    if not metadata_dict:
                        raise RuntimeError(f"LLM Understanding failed: {status_string}")

                    return {
                        "status_message": "Full Hardware Analysis Success",
                        "bpm": metadata_dict.get("bpm"),
                        "keyscale": metadata_dict.get("keyscale"),
                        "timesignature": metadata_dict.get("timesignature"),
                        "duration": metadata_dict.get("duration"),
                        "genre": metadata_dict.get("genres") or metadata_dict.get("genre"),
                        "prompt": metadata_dict.get("caption", ""),
                        "lyrics": metadata_dict.get("lyrics", ""),
                        "language": metadata_dict.get("language", "unknown"),
                        "metas": metadata_dict,
                        "audio_paths": []
                    }

                if req.analysis_only:
                    lm_res = llm_to_pass.generate_with_stop_condition(
                        caption=params.caption,
                        lyrics=params.lyrics,
                        infer_type="dit",
                        temperature=req.lm_temperature,
                        top_p=req.lm_top_p,
                        use_cot_metas=True,
                        use_cot_caption=req.use_cot_caption,
                        use_cot_language=req.use_cot_language,
                        use_constrained_decoding=True
                    )

                    if not lm_res.get("success"):
                        raise RuntimeError(f"Analysis Failed: {lm_res.get('error')}")

                    metas_found = lm_res.get("metadata", {})
                    return {
                        "first_audio_path": None,
                        "audio_paths": [],
                        "raw_audio_paths": [],
                        "generation_info": "Analysis Only Mode Complete",
                        "status_message": "Success",
                        "metas": metas_found,
                        "bpm": metas_found.get("bpm"),
                        "keyscale": metas_found.get("keyscale"),
                        "duration": metas_found.get("duration"),
                        "prompt": metas_found.get("caption", params.caption),
                        "lyrics": params.lyrics,
                        "lm_model": os.getenv("ACESTEP_LM_MODEL_PATH", ""),
                        "dit_model": "None (Analysis Only)"
                    }

                # Generate music using unified interface
                sequential_runs = 1
                if req.task_type == "cover" and h.device == "mps":
                    # If user asked for multiple outputs, run sequentially on MPS to avoid OOM.
                    if config.batch_size is not None and config.batch_size > 1:
                        sequential_runs = int(config.batch_size)
                        config.batch_size = 1
                        print(f"[API Server] Job {job_id}: MPS cover sequential mode enabled (runs={sequential_runs})")

                def _progress_for_slice(start: float, end: float):
                    base = {"seen": False, "value": 0.0}
                    def _cb(value: float, desc: str = "") -> None:
                        try:
                            value_f = max(0.0, min(1.0, float(value)))
                        except Exception:
                            value_f = 0.0
                        if not base["seen"]:
                            base["seen"] = True
                            base["value"] = value_f
                        # Normalize progress to avoid initial jump (e.g., 0.51 -> 0.0)
                        if value_f <= base["value"]:
                            norm = 0.0
                        else:
                            denom = max(1e-6, 1.0 - base["value"])
                            norm = min(1.0, (value_f - base["value"]) / denom)
                        mapped = start + (end - start) * norm
                        _progress_cb(mapped, desc=desc)
                    return _cb

                aggregated_result = None
                all_audios: List[Dict[str, Any]] = []
                for run_idx in range(sequential_runs):
                    if sequential_runs > 1:
                        print(f"[API Server] Job {job_id}: Sequential cover run {run_idx + 1}/{sequential_runs}")
                    if sequential_runs > 1:
                        start = run_idx / sequential_runs
                        end = (run_idx + 1) / sequential_runs
                        progress_cb = _progress_for_slice(start, end)
                    else:
                        progress_cb = _progress_cb

                    result = generate_music(
                        dit_handler=h,
                        llm_handler=llm_to_pass,
                        params=params,
                        config=config,
                        save_dir=app.state.temp_audio_dir,
                        progress=progress_cb,
                    )
                    if not result.success:
                        raise RuntimeError(f"Music generation failed: {result.error or result.status_message}")

                    if aggregated_result is None:
                        aggregated_result = result
                    all_audios.extend(result.audios)

                # Use aggregated result with combined audios
                if aggregated_result is None:
                    raise RuntimeError("Music generation failed: no results")
                aggregated_result.audios = all_audios
                result = aggregated_result

                if not result.success:
                    raise RuntimeError(f"Music generation failed: {result.error or result.status_message}")

                lm_model_name = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
                # Use selected_model_name (set at the beginning of _run_one_job)
                dit_model_name = selected_model_name

                return build_generation_success_response(
                    result=result,
                    params=params,
                    bpm=bpm,
                    audio_duration=audio_duration,
                    key_scale=key_scale,
                    time_signature=time_signature,
                    original_prompt=original_prompt,
                    original_lyrics=original_lyrics,
                    inference_steps=req.inference_steps,
                    path_to_audio_url=_path_to_audio_url,
                    build_generation_info=_build_generation_info,
                    lm_model_name=lm_model_name,
                    dit_model_name=dit_model_name,
                )

            t0 = time.time()
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(executor, _blocking_generate)
                job_store.mark_succeeded(job_id, result)

                # Update local cache
                _update_terminal_job_cache(
                    app_state=app.state,
                    store=store,
                    job_id=job_id,
                    result=result,
                    status="succeeded",
                    map_status=_map_status,
                    result_key_prefix=RESULT_KEY_PREFIX,
                    result_expire_seconds=RESULT_EXPIRE_SECONDS,
                )
            except Exception as e:
                error_traceback = traceback.format_exc()
                print(f"[API Server] Job {job_id} FAILED: {e}")
                print(f"[API Server] Traceback:\n{error_traceback}")
                job_store.mark_failed(job_id, error_traceback)

                # Update local cache
                _update_terminal_job_cache(
                    app_state=app.state,
                    store=store,
                    job_id=job_id,
                    result=None,
                    status="failed",
                    map_status=_map_status,
                    result_key_prefix=RESULT_KEY_PREFIX,
                    result_expire_seconds=RESULT_EXPIRE_SECONDS,
                )
            finally:
                # Best-effort cache cleanup to reduce MPS memory fragmentation between jobs
                try:
                    if hasattr(h, "_empty_cache"):
                        h._empty_cache()
                    else:
                        import torch
                        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()
                except Exception:
                    pass
                dt = max(0.0, time.time() - t0)
                async with app.state.stats_lock:
                    app.state.recent_durations.append(dt)
                    if app.state.recent_durations:
                        app.state.avg_job_seconds = sum(app.state.recent_durations) / len(app.state.recent_durations)

        async def _cleanup_job_temp_files_for_job(job_id: str) -> None:
            await _cleanup_job_temp_files_state(app.state, job_id)

        workers, cleanup_task = start_worker_tasks(
            app_state=app.state,
            store=store,
            worker_count=WORKER_COUNT,
            run_one_job=_run_one_job,
            cleanup_job_temp_files=_cleanup_job_temp_files_for_job,
            cleanup_interval_seconds=JOB_STORE_CLEANUP_INTERVAL,
        )
        initialize_models_at_startup(
            app=app,
            handler=handler,
            llm_handler=llm_handler,
            handler2=handler2,
            handler3=handler3,
            config_path2=config_path2,
            config_path3=config_path3,
            get_project_root=_get_project_root,
            get_model_name=_get_model_name,
            ensure_model_downloaded=_ensure_model_downloaded,
            env_bool=_env_bool,
        )
        try:
            yield
        finally:
            stop_worker_tasks(
                workers=workers,
                cleanup_task=cleanup_task,
                executor=executor,
            )

    app = FastAPI(title="ACE-Step API", version="1.0", lifespan=lifespan)

    configure_api_routes(
        app=app,
        store=store,
        queue_maxsize=QUEUE_MAXSIZE,
        initial_avg_job_seconds=INITIAL_AVG_JOB_SECONDS,
        verify_api_key=verify_api_key,
        verify_token_from_request=verify_token_from_request,
        wrap_response=_wrap_response,
        get_project_root=_get_project_root,
        get_model_name=_get_model_name,
        ensure_model_downloaded=_ensure_model_downloaded,
        env_bool=_env_bool,
        simple_example_data=SIMPLE_EXAMPLE_DATA,
        custom_example_data=CUSTOM_EXAMPLE_DATA,
        format_sample=format_sample,
        to_int=_request_to_int,
        to_float=_request_to_float,
        request_parser_cls=RequestParser,
        request_model_cls=GenerateMusicRequest,
        validate_audio_path=_validate_audio_path,
        save_upload_to_temp=_save_upload_to_temp,
        default_dit_instruction=DEFAULT_DIT_INSTRUCTION,
        lm_default_temperature=LM_DEFAULT_TEMPERATURE,
        lm_default_cfg_scale=LM_DEFAULT_CFG_SCALE,
        lm_default_top_p=LM_DEFAULT_TOP_P,
        map_status=_map_status,
        result_key_prefix=RESULT_KEY_PREFIX,
        task_timeout_seconds=TASK_TIMEOUT_SECONDS,
        log_buffer=log_buffer,
        runtime_start_tensorboard=_runtime_start_tensorboard,
        runtime_stop_tensorboard=_runtime_stop_tensorboard,
        runtime_temporary_llm_model=_runtime_temporary_llm_model,
        runtime_atomic_write_json=_runtime_atomic_write_json,
        runtime_append_jsonl=_runtime_append_jsonl,
    )

    return app


app = create_app()


def main() -> None:
    """CLI entrypoint for API server startup."""

    run_api_server_main(env_bool=_env_bool)

if __name__ == "__main__":
    main()







