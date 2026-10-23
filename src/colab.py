"""Etapa 4: ponte local <-> Google Colab (via pasta do Google Drive sincronizada)."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from .common import ROOT, path_of


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync(cfg: dict) -> None:
    """Zipa o dataset e copia dataset.zip (+ .sha256) para a pasta do Drive."""
    ddir = path_of(cfg, "dataset")
    if not (ddir / "dataset.yaml").exists():
        raise SystemExit("Dataset não encontrado. Rode antes: python pipeline.py build")

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    zpath = dist / "dataset.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ddir / "dataset.yaml", "dataset.yaml")
        for sub in ("images", "labels"):
            for f in sorted((ddir / sub).rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(ddir).as_posix())
    digest = _sha256(zpath)
    (dist / "dataset.zip.sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"dataset.zip: {zpath.stat().st_size / 1e6:.1f} MB  sha256={digest[:12]}...")

    drive = Path(cfg["colab"]["drive_dir"])
    if drive.parent.exists():
        drive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zpath, drive / "dataset.zip")
        shutil.copy2(dist / "dataset.zip.sha256", drive / "dataset.zip.sha256")
        print(f"Copiado para {drive}. O Drive para computador sincroniza sozinho; "
              "espere o ícone indicar que terminou antes de treinar no Colab.")
    else:
        print(f"\nPasta do Drive não encontrada ({drive}).\n"
              f"Ajuste [colab].drive_dir em configs/pipeline.toml ou envie manualmente {zpath} "
              "para a pasta 'yolo_tcc' no seu Google Drive (pelo navegador).")


def fetch(cfg: dict) -> None:
    """Traz de volta runs/<run_name> do Drive e imprime as métricas.
    Sem Drive no PC: usa runs/<run_name> do repo (baixado manualmente)."""
    name = cfg["colab"]["run_name"]
    src = Path(cfg["colab"]["drive_dir"]) / "runs" / name
    dst = ROOT / "runs" / name
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Resultados copiados para {dst}")
    elif dst.exists():
        print(f"Drive não encontrado ({src}); usando a cópia local em {dst}")
    else:
        raise SystemExit(f"Não achei {src} nem {dst}.\n"
                         f"Baixe a pasta runs/{name} do Drive e coloque em {dst}, "
                         "ou ajuste [colab].run_name em configs/pipeline.toml.")

    res = dst / "results.csv"
    if res.exists():
        with open(res, newline="", encoding="utf-8") as f:
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
        if rows:
            def fit(r):
                return 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"])
            b = max(rows, key=fit)
            print(f"\nMelhor época (validação): {b['epoch']} de {rows[-1]['epoch']}  "
                  f"mAP50={float(b['metrics/mAP50(B)']):.3f}  mAP50-95={float(b['metrics/mAP50-95(B)']):.3f}  "
                  f"P={float(b['metrics/precision(B)']):.3f}  R={float(b['metrics/recall(B)']):.3f}")
    tm = dst / "test_metrics.json"
    if tm.exists():
        m = json.loads(tm.read_text(encoding="utf-8"))
        ok = m["mAP50"] >= 0.80
        print(f"\nTESTE: mAP@0.5 = {m['mAP50']:.3f}  ({'APROVADO' if ok else 'REPROVADO'} no limiar 0,80)")
        for k, v in m.get("per_class_mAP50", {}).items():
            print(f"  {k:<24} {v:.3f}")
    print(f"\nPesos: {dst / 'weights' / 'best.pt'}")
