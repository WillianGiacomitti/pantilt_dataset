"""Utilitários compartilhados pela pipeline."""
from __future__ import annotations

import csv
import re
import tomllib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm"}

# Status possíveis de cada frame no manifest
ST_OK = "selecionado"
ST_DARK = "descartado_escuro"
ST_BLUR = "descartado_borrado"
ST_DUP = "descartado_duplicado"
ST_DIV = "descartado_diversidade"

MANIFEST_FIELDS = [
    "frame", "video", "frame_idx", "time_s", "width", "height",
    "blur", "brightness", "status", "batch",
]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "pipeline.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def path_of(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def sanitize(name: str) -> str:
    """Nome seguro para arquivo (sem espaços/acentos), importante no CVAT e no Windows."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return s or "video"


def imwrite(path: Path, img: np.ndarray, quality: int = 95) -> None:
    """cv2.imwrite falha com caminhos não-ASCII no Windows; imencode + write_bytes não."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise OSError(f"Falha ao codificar {path}")
    Path(path).write_bytes(buf.tobytes())


def imread(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


_INT = {"frame_idx", "width", "height"}
_FLOAT = {"time_s", "blur", "brightness"}


def read_manifest(path: Path) -> list[dict]:
    if not Path(path).exists():
        raise SystemExit(f"Manifest não encontrado: {path}. Rode antes: python pipeline.py extract")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in _INT:
                r[k] = int(r[k])
            for k in _FLOAT:
                r[k] = float(r[k])
            rows.append(r)
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
