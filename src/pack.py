"""Etapa 2: monta os zips de imagens (lotes) que serão enviados ao CVAT."""
from __future__ import annotations

import json
import shutil
import zipfile

import cv2

from .common import ST_OK, imread, imwrite, path_of, read_manifest, write_manifest

PALETA = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]

LEIA_ME = """\
COMO USAR NO CVAT (https://app.cvat.ai)
=======================================

1) Crie um Project (uma vez) e, em "Raw", cole o conteudo de cvat_labels.json.
   A ordem das classes define os ids do YOLO (0..{n}); nao reordene depois.

2) Para cada lote_XX.zip: Create new task -> escolha o Project -> "My computer"
   -> envie o zip (o CVAT descompacta e cria um job). Nome da tarefa = nome do zip.

3) Anote com retangulos (atalho N). Regras sugeridas:
   - Anote TODAS as instancias visiveis das 4 classes, inclusive as parciais (>= ~30% visivel).
   - Imagem sem nenhum objeto: deixe em branco (vira imagem de fundo, desejavel ~5%).
   - Mantenha o mesmo criterio de caixa justa em todas as imagens.

4) Terminou um lote: Actions da tarefa -> Export task dataset
   -> formato "YOLO 1.1" -> DESMARQUE "Save images" -> OK. Baixe o zip.

5) Coloque os zips exportados em data/cvat_export/ e rode:
       python pipeline.py build
"""


def run(cfg: dict, batch_size: int | None = None) -> None:
    """Incremental: frames que já têm lote no manifest não mudam (nem o zip deles).
    Só os selecionados ainda sem lote vão para lotes novos, numerados a partir do último."""
    fdir = path_of(cfg, "frames")
    up = path_of(cfg, "cvat_upload")
    mpath = fdir / "manifest.csv"
    rows = read_manifest(mpath)
    sel = [r for r in rows if r["status"] == ST_OK]
    if not sel:
        raise SystemExit("Nenhum frame selecionado no manifest.")

    done = sorted({int(r["batch"]) for r in sel if r["batch"]})
    pending = sorted((r for r in sel if not r["batch"]), key=lambda r: (r["video"], r["frame_idx"]))
    if not pending:
        raise SystemExit(f"Todos os {len(sel)} frames selecionados já estão nos lotes {done}. Nada a empacotar.\n"
                         "Adicione vídeos em data/videos e rode antes: python pipeline.py extract")

    if not done:
        shutil.rmtree(up, ignore_errors=True)  # primeira vez: começa limpo
    up.mkdir(parents=True, exist_ok=True)
    if done:
        print(f"Lotes já existentes (mantidos): {', '.join(f'lote_{b:02d}' for b in done)}")

    size = int(batch_size if batch_size is not None else cfg["cvat"]["batch_size"])
    if size <= 0:
        raise SystemExit("batch_size precisa ser > 0.")
    max_side = int(cfg["cvat"].get("max_side", 0) or 0)
    q = cfg["extract"]["jpeg_quality"]
    batches = [pending[i:i + size] for i in range(0, len(pending), size)]
    first = (done[-1] + 1) if done else 1

    for b, chunk in enumerate(batches, first):
        zpath = up / f"lote_{b:02d}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for r in chunk:
                src = fdir / "selected" / f"{r['frame']}.jpg"
                r["batch"] = b
                if max_side > 0:
                    img = imread(src)
                    h, w = img.shape[:2]
                    s = max_side / max(h, w)
                    if s < 1:
                        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                    tmp = up / "_tmp.jpg"
                    imwrite(tmp, img, q)
                    z.write(tmp, arcname=f"{r['frame']}.jpg")
                    tmp.unlink()
                else:
                    z.write(src, arcname=f"{r['frame']}.jpg")
        print(f"{zpath.name}: {len(chunk)} imagens ({zpath.stat().st_size / 1e6:.1f} MB)")

    classes = cfg["classes"]
    labels = [
        {"name": c, "color": PALETA[i % len(PALETA)], "type": "rectangle", "attributes": []}
        for i, c in enumerate(classes)
    ]
    (up / "cvat_labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    (up / "LEIA-ME_CVAT.txt").write_text(LEIA_ME.format(n=len(classes) - 1), encoding="utf-8")
    write_manifest(mpath, rows)  # grava a coluna "batch"
    print(f"\n{len(batches)} lote(s) novo(s) em {up}\nLeia {up / 'LEIA-ME_CVAT.txt'}")
