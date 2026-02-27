"""Unit tests for audio route registration."""

import asyncio
import unittest
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute

from acestep.api.http.audio_route import register_audio_route


async def _verify_api_key(_authorization: str | None = None) -> None:
    """No-op auth dependency for unit-level route execution."""

    return None


def _get_endpoint(app: FastAPI, path: str, method: str):
    """Return endpoint callable matching route path and method."""

    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path and method.upper() in route.methods:
            return route.endpoint
    raise AssertionError(f"Missing route: {method} {path}")


class AudioRouteTests(unittest.TestCase):
    """Behavior tests for ``GET /v1/audio`` security and file checks."""

    def test_rejects_path_outside_allowed_dir(self):
        """Audio route should return HTTP 403 for paths outside allowed directory."""

        app = FastAPI()
        app.state.temp_audio_dir = str(Path.cwd())
        register_audio_route(app=app, verify_api_key=_verify_api_key)
        endpoint = _get_endpoint(app, "/v1/audio", "GET")
        request = type("Req", (), {"app": app})()

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(endpoint(path=str(Path("C:/outside.mp3")), request=request, _=None))
        self.assertEqual(403, ctx.exception.status_code)

    def test_returns_404_when_file_missing(self):
        """Audio route should return HTTP 404 for missing files inside allowed dir."""

        app = FastAPI()
        app.state.temp_audio_dir = str(Path.cwd())
        register_audio_route(app=app, verify_api_key=_verify_api_key)
        endpoint = _get_endpoint(app, "/v1/audio", "GET")
        request = type("Req", (), {"app": app})()

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(endpoint(path=str(Path.cwd() / "missing.mp3"), request=request, _=None))
        self.assertEqual(404, ctx.exception.status_code)


if __name__ == "__main__":
    unittest.main()
