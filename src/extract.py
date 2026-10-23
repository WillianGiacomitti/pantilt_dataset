"""Etapa 1: vídeos -> frames diversos e nítidos.

Cascata de filtros (do mais barato ao mais caro):
  1. amostragem temporal (sample_fps)
  2. descarte de frames escuros
  3. descarte de frames borrados (variância do Laplaciano, relativa ao percentil 75 do vídeo)
  4. deduplicação em sequência (dHash): cada "corrida" de frames parecidos mantém só o mais nítido
  5. seleção por diversidade global (farthest-point sampling) entre todos os vídeos

Incremental: com manifest existente, só vídeos novos são lidos, e a seleção de diversidade
evita frames parecidos com os já selecionados.
"""
from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import (
    ST_BLUR, ST_DARK, ST_DIV, ST_DUP, ST_OK, VIDEO_EXTS,
    imread, imwrite, path_of, read_manifest, sanitize, write_manifest,
)


# ----------------------------------------------------------------------------
# Descritores
# ----------------------------------------------------------------------------
def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _blur_score(gray: np.ndarray) -> float:
    """Variância do Laplaciano em largura fixa (640 px) para ser comparável entre resoluções."""
    h, w = gray.shape
    if w != 640:
        gray = cv2.resize(gray, (640, max(1, round(h * 640 / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _feature(bgr: np.ndarray) -> np.ndarray:
    """Vetor de diversidade: miniatura LAB (layout + iluminação) + histograma HSV (cores)."""
    thumb = cv2.resize(bgr, (32, 18), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(thumb, cv2.COLOR_BGR2LAB).astype(np.float32).ravel() / 255.0
    lab = lab / np.sqrt(lab.size) * 3.0  # ~ RMS por pixel, ponderado
    small = cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel().astype(np.float32)
    hist = np.sqrt(hist / (hist.sum() + 1e-8))
    hist = hist / (np.linalg.norm(hist) + 1e-8) * 0.5
    return np.concatenate([lab, hist])


# ----------------------------------------------------------------------------
# Passo 1-2: leitura do vídeo
# ----------------------------------------------------------------------------
def _sample_video(video: Path, stem: str, ex: dict, out_dir: Path) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(
            f"Não consegui abrir {video}. Se o nome tem acentos/caracteres especiais, renomeie o arquivo."
        )
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    step = max(1, round(fps / ex["sample_fps"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, idx = [], 0
    while cap.grab():
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                name = f"{stem}_f{idx:06d}"
                imwrite(out_dir / f"{name}.jpg", frame, ex["jpeg_quality"])
                h, w = frame.shape[:2]
                rows.append({
                    "frame": name, "video": stem, "frame_idx": idx,
                    "time_s": round(idx / fps, 3), "width": w, "height": h,
                    "blur": round(_blur_score(gray), 2),
                    "brightness": round(float(gray.mean()), 1),
                    "status": "candidato", "batch": "",
                    "_hash": _dhash(gray), "_feat": _feature(frame),
                })
        idx += 1
    cap.release()
    return rows


# ----------------------------------------------------------------------------
# Passo 3-4: filtros por vídeo
# ----------------------------------------------------------------------------
def _filter_video(rows: list[dict], ex: dict) -> None:
    if not rows:
        return
    for r in rows:
        if r["brightness"] < ex["min_brightness"]:
            r["status"] = ST_DARK

    alive = [r for r in rows if r["status"] == "candidato"]
    if alive:
        ref = float(np.percentile([r["blur"] for r in alive], 75))  # nitidez "típica" dos frames bons
        limit = max(ex["blur_abs_min"], ex["blur_rel_threshold"] * ref)
        for r in alive:
            if r["blur"] < limit:
                r["status"] = ST_BLUR

    alive = [r for r in rows if r["status"] == "candidato"]
    runs, cur, anchor = [], [], None
    for r in alive:
        if anchor is not None and (r["_hash"] ^ anchor["_hash"]).bit_count() <= ex["hash_threshold"]:
            cur.append(r)
        else:
            if cur:
                runs.append(cur)
            cur, anchor = [r], r
    if cur:
        runs.append(cur)
    for run in runs:
        best = max(run, key=lambda r: r["blur"])
        for r in run:
            if r is not best:
                r["status"] = ST_DUP


# ----------------------------------------------------------------------------
# Passo 5: diversidade global
# ----------------------------------------------------------------------------
def farthest_point_sampling(
    X: np.ndarray, k: int, min_dist: float, seed: np.ndarray | None = None,
) -> tuple[list[int], float]:
    """Escolhe até k pontos bem espalhados. Para se o ponto mais distinto restante estiver
    a menos de min_dist dos já escolhidos. Retorna (índices, distância em que parou).
    seed = descritores já escolhidos antes (extração incremental): os novos pontos
    precisam ser distintos deles também."""
    n = len(X)
    if n == 0 or k <= 0:
        return [], 0.0
    if seed is not None and len(seed):
        sel = []
        d = np.full(n, np.inf)
        for s in seed:
            d = np.minimum(d, np.linalg.norm(X - s, axis=1))
    else:
        first = int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))
        sel = [first]
        d = np.linalg.norm(X - X[first], axis=1)
    stop_d = float("inf")
    while len(sel) < min(k, n):
        j = int(np.argmax(d))
        stop_d = float(d[j])
        if stop_d < min_dist:
            break
        sel.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return sel, stop_d


# ----------------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------------
def _video_stems(videos: list[Path]) -> list[tuple[Path, str]]:
    """Nome (stem) de cada vídeo no manifest. Determinístico: a mesma pasta gera os mesmos nomes."""
    out, used = [], set()
    for v in videos:
        stem, n = sanitize(v.stem), 2
        base = stem
        while stem in used:
            stem, n = f"{base}_{n}", n + 1
        used.add(stem)
        out.append((v, stem))
    return out


def _selected_features(rows: list[dict], selected: Path) -> np.ndarray | None:
    feats = []
    for r in rows:
        if r["status"] != ST_OK:
            continue
        img = imread(selected / f"{r['frame']}.jpg")
        if img is None:
            raise SystemExit(f"Frame do manifest não encontrado: {selected / r['frame']}.jpg")
        feats.append(_feature(img))
    return np.stack(feats) if feats else None


def run(cfg: dict, force: bool = False, max_frames: int | None = None) -> None:
    """Sem --force e com manifest existente: modo incremental. Só os vídeos que ainda não estão
    no manifest são processados; frames, status e lotes antigos ficam intactos."""
    ex = cfg["extract"]
    cap = int(max_frames if max_frames is not None else ex["max_frames"])
    vdir, fdir = path_of(cfg, "videos"), path_of(cfg, "frames")
    rdir = path_of(cfg, "reports")
    sampled, selected, rejected = fdir / "_sampled", fdir / "selected", fdir / "rejected"
    mpath = fdir / "manifest.csv"
    incremental = mpath.exists() and not force

    if not incremental and not force and selected.exists() and any(selected.iterdir()):
        raise SystemExit(
            "Há frames em data/frames/selected mas não há manifest.csv.\n"
            "Para refazer tudo do zero, rode: python pipeline.py extract --force"
        )

    videos = sorted(p for p in vdir.glob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"Nenhum vídeo encontrado em {vdir} (extensões: {sorted(VIDEO_EXTS)})")

    old_rows = read_manifest(mpath) if incremental else []
    known = {r["video"] for r in old_rows}
    todo = [(v, s) for v, s in _video_stems(videos) if s not in known]
    if incremental:
        if not todo:
            raise SystemExit(
                f"Nenhum vídeo novo: os {len(known)} vídeo(s) de {vdir} já estão no manifest.\n"
                "Para refazer tudo (apaga seleção e lotes!), rode: python pipeline.py extract --force"
            )
        print(f"Modo incremental: {len(known)} vídeo(s) já no manifest ficam intactos; "
              f"{len(todo)} vídeo(s) novo(s).", flush=True)

    shutil.rmtree(sampled, ignore_errors=True)
    if not incremental:
        for d in (selected, rejected):
            shutil.rmtree(d, ignore_errors=True)
    selected.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)

    seed = None
    if incremental:
        print("Calculando descritores dos frames já selecionados ...", flush=True)
        seed = _selected_features(old_rows, selected)

    all_rows = []
    for v, stem in todo:
        print(f"[{stem}] lendo {v.name} ...", flush=True)
        rows = _sample_video(v, stem, ex, sampled / stem)
        _filter_video(rows, ex)
        all_rows.extend(rows)

    # diversidade global (no incremental, também em relação aos frames já selecionados)
    alive = [r for r in all_rows if r["status"] == "candidato"]
    stop_d = 0.0
    if alive:
        X = np.stack([r["_feat"] for r in alive])
        sel_idx, stop_d = farthest_point_sampling(X, cap, ex["feature_min_distance"], seed)
        chosen = set(sel_idx)
        for i, r in enumerate(alive):
            r["status"] = ST_OK if i in chosen else ST_DIV

    # materializa arquivos
    for r in all_rows:
        src = sampled / r["video"] / f"{r['frame']}.jpg"
        if r["status"] == ST_OK:
            shutil.move(src, selected / f"{r['frame']}.jpg")
        elif ex["keep_rejected"]:
            dst = rejected / r["status"]
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst / f"{r['frame']}.jpg")
    shutil.rmtree(sampled, ignore_errors=True)

    if incremental:
        n_sel_old = sum(r["status"] == ST_OK for r in old_rows)
        print(f"\nFrames já selecionados antes (mantidos): {n_sel_old}")
    all_rows = old_rows + all_rows
    all_rows.sort(key=lambda r: (r["video"], r["frame_idx"]))
    write_manifest(mpath, all_rows)
    new_sel = sum(r["status"] == ST_OK for r in all_rows) - (n_sel_old if incremental else 0)
    _summary(all_rows, rdir / "extracao.csv", stop_d, ex, new_sel, cap)


def _summary(rows: list[dict], out_csv: Path, stop_d: float, ex: dict, new_sel: int, cap: int) -> None:
    per = {}
    for r in rows:
        per.setdefault(r["video"], Counter())[r["status"]] += 1
    cols = [ST_OK, ST_DARK, ST_BLUR, ST_DUP, ST_DIV]
    head = ["video", "amostrados"] + cols
    print("\n" + " | ".join(f"{h:>22}" if i else f"{h:<28}" for i, h in enumerate(head)))
    tot = Counter()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        for v, c in per.items():
            line = [v, sum(c.values())] + [c[k] for k in cols]
            w.writerow(line)
            tot.update(c)
            print(" | ".join(f"{x:>22}" if i else f"{x:<28}" for i, x in enumerate(line)))
    for v, c in per.items():
        n = sum(c.values())
        if n and c[ST_BLUR] / n > 0.7:
            print(f"[aviso] {v}: {c[ST_BLUR] / n:.0%} dos frames descartados por borrão. "
                  "Veja blur no manifest.csv e reduza blur_rel_threshold (ou o vídeo está mesmo todo em movimento).")
    print(f"\nTotal amostrado: {sum(tot.values())}  ->  selecionado: {tot[ST_OK]} "
          f"({new_sel} novo(s) nesta execução)")
    if new_sel >= cap:
        print(f"Atingiu o teto max_frames={cap} desta execução.")
    else:
        print(f"Seleção de diversidade parou em distância {stop_d:.3f} "
              f"(feature_min_distance={ex['feature_min_distance']}).")
    print("Ajuste os limiares em configs/pipeline.toml (ou --max-frames) se quiser mais/menos frames.")
    print(f"Resumo salvo em {out_csv}")
