"""HTTP route for serving generated audio files by path."""

from __future__ import annotations

import os
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request


def register_audio_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
) -> None:
    """Register the ``GET /v1/audio`` route."""

    @app.get("/v1/audio")
    async def get_audio(path: str, request: Request, _: None = Depends(verify_api_key)):
        """Serve a generated audio file when path is within the allowed directory."""

        from fastapi.responses import FileResponse

        resolved_path = os.path.realpath(path)
        allowed_dir = os.path.realpath(request.app.state.temp_audio_dir)
        if not resolved_path.startswith(allowed_dir + os.sep) and resolved_path != allowed_dir:
            raise HTTPException(status_code=403, detail="Access denied: path outside allowed directory")
        if not os.path.exists(resolved_path):
            raise HTTPException(status_code=404, detail="Audio file not found")

        ext = os.path.splitext(resolved_path)[1].lower()
        media_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        return FileResponse(resolved_path, media_type=media_types.get(ext, "audio/mpeg"))
