"""Etapa 3: exportação do CVAT (YOLO 1.1) -> validação -> split por vídeo -> dataset final."""
from __future__ import annotations

import csv
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

from .common import ST_OK, path_of, read_manifest

IGNORE_STEMS = {"train", "valid", "val", "test", "obj", "classes", "data"}


# ----------------------------------------------------------------------------
# Leitura dos exports do CVAT
# ----------------------------------------------------------------------------
def _read_sources(export_dir: Path) -> list[tuple[str, dict[str, bytes]]]:
    """Cada fonte = um zip (ou pasta já extraída). Retorna {caminho_no_zip: bytes} dos .txt/.names/.yaml."""
    sources = []
    wanted = (".txt", ".names", ".yaml", ".yml")
    for z in sorted(export_dir.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            files = {i.filename: zf.read(i) for i in zf.infolist()
                     if not i.is_dir() and i.filename.lower().endswith(wanted)}
        sources.append((z.name, files))
    for d in sorted(p for p in export_dir.iterdir() if p.is_dir()):
        files = {str(f.relative_to(d)).replace("\\", "/"): f.read_bytes()
                 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in wanted}
        sources.append((d.name, files))
    return sources


def _source_names(files: dict[str, bytes]) -> list[str] | None:
    for path, data in files.items():
        if path.lower().endswith("obj.names") or Path(path).name.lower() in ("classes.txt", "data.names"):
            names = [ln.strip() for ln in data.decode("utf-8-sig").splitlines() if ln.strip()]
            if names:
                return names
    for path, data in files.items():
        if Path(path).name.lower() in ("data.yaml", "dataset.yaml"):
            pairs = re.findall(r"^\s+(\d+):\s*(.+?)\s*$", data.decode("utf-8-sig"), flags=re.M)
            if pairs:
                return [n.strip("'\"") for _, n in sorted(pairs, key=lambda p: int(p[0]))]
    return None


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def _parse_label(text: str, names: list[str], class_ids: dict[str, int], frame: str, issues: list[dict]):
    boxes = []

    def issue(level, line_no, msg):
        issues.append({"nivel": level, "frame": frame, "linha": line_no, "problema": msg})

    for ln, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            issue("erro", ln, f"esperados 5 valores, achei {len(parts)} (polígono/segmentação não é suportado)")
            continue
        try:
            c = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            issue("erro", ln, "valores não numéricos")
            continue
        if not 0 <= c < len(names):
            issue("erro", ln, f"id de classe {c} fora da lista do export ({len(names)} classes)")
            continue
        if names[c] not in class_ids:
            issue("erro", ln, f"classe '{names[c]}' não existe em configs/pipeline.toml")
            continue
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
        if max(abs(cx1 - x1), abs(cy1 - y1), abs(cx2 - x2), abs(cy2 - y2)) > 1e-3:
            issue("aviso", ln, "caixa saía da imagem; foi recortada")
        w, h = cx2 - cx1, cy2 - cy1
        if w <= 0 or h <= 0 or w * h < 1e-6:
            issue("aviso", ln, "caixa degenerada (área ~0); removida")
            continue
        box = (class_ids[names[c]], (cx1 + cx2) / 2, (cy1 + cy2) / 2, w, h)
        if any(b[0] == box[0] and _iou(b, box) > 0.95 for b in boxes):
            issue("aviso", ln, "caixa duplicada (IoU > 0.95, mesma classe); removida")
            continue
        boxes.append(box)
    return boxes


def load_annotations(cfg: dict, manifest: dict[str, dict]):
    """Retorna (labels {frame: [boxes]}, frames_excluidos_sem_anotacao, issues)."""
    class_ids = {c: i for i, c in enumerate(cfg["classes"])}
    export_dir = path_of(cfg, "cvat_export")
    export_dir.mkdir(parents=True, exist_ok=True)
    sources = _read_sources(export_dir)
    if not sources:
        raise SystemExit(f"Nenhum zip do CVAT em {export_dir}. Exporte como 'YOLO 1.1' e coloque lá.")

    labels, issues, origin = {}, [], {}
    for src, files in sources:
        names = _source_names(files)
        if names is None:
            print(f"[aviso] {src}: sem obj.names/data.yaml; assumindo a ordem de configs/pipeline.toml")
            names = list(cfg["classes"])
        elif names != cfg["classes"]:
            print(f"[aviso] {src}: ordem/nomes das classes diferem do config; remapeando por NOME.")
        n_ok = 0
        for path, data in files.items():
            stem = Path(path).stem
            if not path.lower().endswith(".txt") or stem.lower() in IGNORE_STEMS:
                continue
            if stem not in manifest:
                issues.append({"nivel": "aviso", "frame": stem, "linha": 0,
                               "problema": f"{src}: label sem imagem correspondente no manifest (ignorado)"})
                continue
            if stem in labels:
                issues.append({"nivel": "aviso", "frame": stem, "linha": 0,
                               "problema": f"anotado em mais de um export ({origin[stem]} e {src}); usei o último"})
            labels[stem] = _parse_label(data.decode("utf-8-sig"), names, class_ids, stem, issues)
            origin[stem] = src
            n_ok += 1
        print(f"{src}: {n_ok} arquivos de label lidos")

    # Lote exportado = lote com pelo menos 1 label. Frame sem .txt num lote exportado = fundo.
    exported_batches = {manifest[f]["batch"] or "0" for f in labels}
    excluded = []
    for f, r in manifest.items():
        if f in labels:
            continue
        if (r["batch"] or "0") in exported_batches:
            labels[f] = []  # imagem de fundo
        else:
            excluded.append(f)
    return labels, excluded, issues


# ----------------------------------------------------------------------------
# Split por vídeo
# ----------------------------------------------------------------------------
def split_videos(videos: list[str], A: np.ndarray, I: np.ndarray, ratios: dict[str, float],
                 seed: int, iters: int) -> dict[str, str]:
    """Atribui cada VÍDEO inteiro a um split, minimizando o desvio das proporções de imagens e de
    instâncias por classe. Busca aleatória vetorizada."""
    names = list(ratios)
    r = np.array([ratios[n] for n in names], dtype=float)
    r = r / r.sum()
    n = len(videos)
    if n < len(names):
        raise SystemExit(
            f"Só há {n} vídeo(s) anotado(s) para {len(names)} splits. Grave/anote mais vídeos "
            "(o split é por vídeo para evitar vazamento entre treino e validação)."
        )
    rng = np.random.default_rng(seed)
    assign = rng.choice(len(names), size=(iters, n), p=r)
    totA, totI = A.sum(), I.sum(0)
    present = totI > 0
    score = np.zeros(iters)
    for s in range(len(names)):
        mask = (assign == s).astype(float)
        a, i = mask @ A, mask @ I
        score += np.abs(a - r[s] * totA) / max(totA, 1)
        if present.any():
            score += (np.abs(i - r[s] * totI)[:, present] / totI[present]).mean(1)
            score += 2.0 * (i[:, present] == 0).sum(1)  # classe sumiu deste split
        score += 10.0 * (a == 0)
    best = assign[int(np.argmin(score))]
    return {v: names[best[k]] for k, v in enumerate(videos)}


# ----------------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------------
def run(cfg: dict, allow_issues: bool = False) -> None:
    from . import report

    fdir, ddir, rdir = path_of(cfg, "frames"), path_of(cfg, "dataset"), path_of(cfg, "reports")
    rdir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_manifest(fdir / "manifest.csv") if r["status"] == ST_OK]
    manifest = {r["frame"]: r for r in rows}

    labels, excluded, issues = load_annotations(cfg, manifest)
    with open(rdir / "problemas_anotacao.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["nivel", "frame", "linha", "problema"])
        w.writeheader()
        w.writerows(issues)
    errors = [i for i in issues if i["nivel"] == "erro"]
    warns = [i for i in issues if i["nivel"] == "aviso"]
    print(f"\nValidação: {len(errors)} erro(s), {len(warns)} aviso(s) -> {rdir / 'problemas_anotacao.csv'}")
    if errors and not allow_issues:
        for e in errors[:10]:
            print(f"  ERRO {e['frame']} linha {e['linha']}: {e['problema']}")
        raise SystemExit("Corrija no CVAT e exporte de novo (ou use --allow-issues para ignorar as linhas com erro).")
    if excluded:
        print(f"{len(excluded)} imagem(ns) de lotes ainda não exportados foram EXCLUÍDAS do dataset.")
    if not labels:
        raise SystemExit("Nenhuma imagem anotada.")

    # agrupa por vídeo
    n_cls = len(cfg["classes"])
    vids = sorted({manifest[f]["video"] for f in labels})
    A = np.zeros(len(vids))
    I = np.zeros((len(vids), n_cls))
    vidx = {v: k for k, v in enumerate(vids)}
    for f, boxes in labels.items():
        k = vidx[manifest[f]["video"]]
        A[k] += 1
        for b in boxes:
            I[k, b[0]] += 1
    if len(vids) < 10:
        print(f"[aviso] só {len(vids)} vídeos: o equilíbrio entre splits pode ficar imperfeito (veja o relatório).")

    sp = cfg["split"]
    video_split = split_videos(vids, A, I, sp["ratios"], sp["seed"], sp["iterations"])

    # grava dataset
    shutil.rmtree(ddir, ignore_errors=True)
    for s in sp["ratios"]:
        (ddir / "images" / s).mkdir(parents=True)
        (ddir / "labels" / s).mkdir(parents=True)
    split_rows = []
    for f, boxes in sorted(labels.items()):
        s = video_split[manifest[f]["video"]]
        src = fdir / "selected" / f"{f}.jpg"
        if not src.exists():
            raise SystemExit(f"Imagem ausente: {src}")
        shutil.copy2(src, ddir / "images" / s / f"{f}.jpg")
        txt = "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes)
        (ddir / "labels" / s / f"{f}.txt").write_text(txt + ("\n" if txt else ""), encoding="utf-8")
        split_rows.append({"frame": f, "video": manifest[f]["video"], "split": s, "n_boxes": len(boxes)})
    with open(ddir / "splits.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "video", "split", "n_boxes"])
        w.writeheader()
        w.writerows(split_rows)

    yaml = ["# Gerado por pipeline.py build", "train: images/train", "val: images/val"]
    if "test" in sp["ratios"]:
        yaml.append("test: images/test")
    yaml += ["names:"] + [f"  {i}: {c}" for i, c in enumerate(cfg["classes"])]
    (ddir / "dataset.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"\nDataset gravado em {ddir}")
    for s in sp["ratios"]:
        vs = [v for v, x in video_split.items() if x == s]
        n_img = sum(1 for r in split_rows if r["split"] == s)
        print(f"  {s:<5}: {n_img:>4} imagens, {len(vs)} vídeo(s)")
    report.generate(cfg, excluded=excluded, issues=issues)
