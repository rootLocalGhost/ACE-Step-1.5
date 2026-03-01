"""Service helpers for `/query_result` response shaping."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List


def parse_task_id_list(task_id_list_raw: Any) -> List[str]:
    """Parse a raw ``task_id_list`` field value into task IDs.

    Args:
        task_id_list_raw: Either a list of IDs or JSON-encoded list text.

    Returns:
        Parsed task IDs, or an empty list when parsing fails.
    """

    if isinstance(task_id_list_raw, list):
        return task_id_list_raw
    try:
        parsed = json.loads(task_id_list_raw)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _build_running_result_payload(
    task_id: str,
    data: str,
    current_time: float,
    task_timeout_seconds: int,
    log_last_message: str,
) -> Dict[str, Any]:
    """Return one cache-derived response item for the given task."""

    try:
        data_json = json.loads(data)
    except Exception:
        data_json = []

    if len(data_json) <= 0:
        return {"task_id": task_id, "result": data, "status": 2}

    status = data_json[0].get("status")
    create_time = data_json[0].get("create_time", 0)
    if status == 0 and (current_time - create_time) > task_timeout_seconds:
        return {"task_id": task_id, "result": data, "status": 2}
    return {
        "task_id": task_id,
        "result": data,
        "status": int(status) if status is not None else 1,
        "progress_text": log_last_message,
    }


def _build_store_result_payload(
    task_id: str,
    record: Any,
    map_status: Callable[[str], int],
    log_last_message: str,
) -> Dict[str, Any]:
    """Return one store-derived response item for the given task."""

    env = getattr(record, "env", "development")
    create_time = record.created_at
    status_int = map_status(record.status)

    if record.result and record.status == "succeeded":
        if record.result.get("status_message") == "Full Hardware Analysis Success":
            result_data = [record.result]
        else:
            audio_paths = record.result.get("audio_paths", [])
            metas = record.result.get("metas", {}) or {}
            result_data = [
                {
                    "file": path,
                    "wave": "",
                    "status": status_int,
                    "create_time": int(create_time),
                    "env": env,
                    "prompt": metas.get("caption", ""),
                    "lyrics": metas.get("lyrics", ""),
                    "metas": {
                        "bpm": metas.get("bpm"),
                        "duration": metas.get("duration"),
                        "genres": metas.get("genres", ""),
                        "keyscale": metas.get("keyscale", ""),
                        "timesignature": metas.get("timesignature", ""),
                    },
                }
                for path in audio_paths
            ] if audio_paths else [{
                "file": "",
                "wave": "",
                "status": status_int,
                "create_time": int(create_time),
                "env": env,
                "prompt": metas.get("caption", ""),
                "lyrics": metas.get("lyrics", ""),
                "metas": {
                    "bpm": metas.get("bpm"),
                    "duration": metas.get("duration"),
                    "genres": metas.get("genres", ""),
                    "keyscale": metas.get("keyscale", ""),
                    "timesignature": metas.get("timesignature", ""),
                },
            }]
    else:
        result_data = [{
            "file": "",
            "wave": "",
            "status": status_int,
            "create_time": int(create_time),
            "env": env,
            "prompt": "",
            "lyrics": "",
            "metas": {},
            "progress": float(record.progress) if record else 0.0,
            "stage": record.stage if record else "queued",
            "error": record.error if record.error else None,
        }]

    current_log = log_last_message if status_int == 0 else record.progress_text
    return {
        "task_id": task_id,
        "result": json.dumps(result_data, ensure_ascii=False),
        "status": status_int,
        "progress_text": current_log,
    }


def collect_query_results(
    task_ids: List[str],
    local_cache: Any,
    store: Any,
    map_status: Callable[[str], int],
    result_key_prefix: str,
    task_timeout_seconds: int,
    log_last_message: str,
) -> List[Dict[str, Any]]:
    """Collect legacy response items for each requested task ID.

    Args:
        task_ids: Task IDs to query.
        local_cache: Cache object exposing ``get(key)``.
        store: In-memory store exposing ``get(task_id)``.
        map_status: Mapper from store status text to integer code.
        result_key_prefix: Prefix used for local-cache lookup keys.
        task_timeout_seconds: Timeout threshold for running cached jobs.
        log_last_message: Current global progress log text.

    Returns:
        Ordered list of response items matching ``/query_result`` contract.
    """

    current_time = time.time()
    data_list = []
    for task_id in task_ids:
        result_key = f"{result_key_prefix}{task_id}"
        if local_cache:
            data = local_cache.get(result_key)
            if data:
                data_list.append(
                    _build_running_result_payload(
                        task_id=task_id,
                        data=data,
                        current_time=current_time,
                        task_timeout_seconds=task_timeout_seconds,
                        log_last_message=log_last_message,
                    )
                )
                continue

        record = store.get(task_id)
        if record:
            data_list.append(
                _build_store_result_payload(
                    task_id=task_id,
                    record=record,
                    map_status=map_status,
                    log_last_message=log_last_message,
                )
            )
        else:
            data_list.append({"task_id": task_id, "result": "[]", "status": 0})
    return data_list
