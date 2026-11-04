from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .config import PipelineConfig
from .engine import FaceEngine
from .pipeline import identify

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_POSTER_BYTES = 20 * 1024 * 1024
MAX_PORTRAIT_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = PipelineConfig()
    app.state.config = config
    app.state.engine = FaceEngine(config)
    yield


app = FastAPI(
    title="facedec",
    version="0.1.0",
    description="CPU-only poster cast identification feasibility API",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "model": app.state.config.model_name,
        "providers": ["CPUExecutionProvider"],
    }


def _safe_name(upload: UploadFile) -> str:
    name = Path(upload.filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="upload is missing a filename")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported image suffix: {suffix or '(none)'}",
        )
    return name


async def _save_upload(upload: UploadFile, path: Path, max_bytes: int) -> None:
    data = await upload.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{upload.filename} exceeds {max_bytes} bytes",
        )
    path.write_bytes(data)


@app.post("/v1/identify")
async def identify_endpoint(
    poster: UploadFile = File(...),
    cast: str = Form(..., description="cast JSON with portrait_files per member"),
    portraits: list[UploadFile] = File(...),
) -> dict:
    """Identify uploaded poster faces against uploaded cast portraits.

    ``cast`` example::

        {"members": [
          {"cast_id": "c1", "name": "Actor A", "portrait_files": ["a.jpg"]}
        ]}
    """

    try:
        cast_payload = json.loads(cast)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid cast JSON: {exc}") from exc

    members = cast_payload.get("members") if isinstance(cast_payload, dict) else None
    if not isinstance(members, list) or not members:
        raise HTTPException(status_code=400, detail="cast.members must be non-empty")

    temp_dir = Path(tempfile.mkdtemp(prefix="facedec_"))
    try:
        poster_dir = temp_dir / "poster"
        portrait_dir = temp_dir / "portraits"
        poster_dir.mkdir()
        portrait_dir.mkdir()

        poster_name = _safe_name(poster)
        poster_path = poster_dir / poster_name
        await _save_upload(poster, poster_path, MAX_POSTER_BYTES)

        file_map: dict[str, Path] = {}
        for upload in portraits:
            name = _safe_name(upload)
            if name in file_map:
                raise HTTPException(
                    status_code=400,
                    detail=f"duplicate portrait filename: {name}",
                )
            path = portrait_dir / name
            await _save_upload(upload, path, MAX_PORTRAIT_BYTES)
            file_map[name] = path

        for member in members:
            if not isinstance(member, dict):
                raise HTTPException(
                    status_code=400,
                    detail="each member must be an object",
                )
            names = member.get("portrait_files")
            if not isinstance(names, list) or not names:
                raise HTTPException(
                    status_code=400,
                    detail="each member needs a non-empty portrait_files list",
                )
            missing = [name for name in names if name not in file_map]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"missing uploaded portraits: {missing}",
                )
            member["portrait_paths"] = [str(file_map[name]) for name in names]

        try:
            result = await run_in_threadpool(
                identify,
                poster_path,
                cast_payload,
                app.state.engine,
                app.state.config,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("facedec.api:app", host="127.0.0.1", port=8000, reload=False)
