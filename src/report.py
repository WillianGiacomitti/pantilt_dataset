"""Relatório da composição do dataset anotado (HTML autocontido + PNGs + CSVs)."""
from __future__ import annotations

import base64
import csv
import html
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import imread, path_of, read_manifest  # noqa: E402

SPLIT_COLORS = {"train": "#3b6ea5", "val": "#d98e3f", "test": "#5a9e6f"}
CLASS_COLORS = ["#c0504d", "#4f81bd", "#8064a2", "#e0a030", "#4bacc6", "#7f7f7f"]
COCO_SMALL, COCO_MED = 32, 96  # lado equivalente em px (sqrt da área)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100,
})


# ----------------------------------------------------------------------------
# Coleta
# ----------------------------------------------------------------------------
def _collect(cfg: dict):
    ddir, fdir = path_of(cfg, "dataset"), path_of(cfg, "frames")
    man = {r["frame"]: r for r in read_manifest(fdir / "manifest.csv")}
    imgs = []
    with open(ddir / "splits.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab = ddir / "labels" / r["split"] / f"{r['frame']}.txt"
            boxes = []
            for line in lab.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5:
                    boxes.append((int(p[0]), *map(float, p[1:])))
            m = man[r["frame"]]
            imgs.append({**r, "boxes": boxes, "W": m["width"], "H": m["height"],
                         "blur": m["blur"], "brightness": m["brightness"]})
    return imgs


def _b64_png(fig, png_path: Path) -> str:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------
def _fig_instances(classes, splits, inst, min_inst):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    series = [(s, [inst[s][c] for c in range(len(classes))], SPLIT_COLORS.get(s, "#888")) for s in splits]
    series.append(("total", [sum(inst[s][c] for s in splits) for c in range(len(classes))], "#555555"))
    n = len(series)
    wid = 0.84 / n
    for k, (name, vals, color) in enumerate(series):
        bars = ax.bar(np.arange(len(classes)) + k * wid - 0.42 + wid / 2, vals, wid, label=name, color=color)
        ax.bar_label(bars, fontsize=8, padding=1)
    ax.axhline(min_inst, color="#444", ls="--", lw=1)
    ax.text(-0.45, min_inst, f"meta: {min_inst} instâncias no total por classe ", va="bottom", ha="left", fontsize=8)
    ax.set_xticks(range(len(classes)), classes, rotation=15)
    ax.set_ylabel("instâncias (caixas)")
    ax.set_title("Instâncias por classe e por split", loc="left", pad=22)
    ax.legend(frameon=False, ncol=n, loc="lower right", bbox_to_anchor=(1, 1.0))
    return fig


def _fig_videos(per_video):
    vids = sorted(per_video, key=lambda v: (per_video[v]["split"], v))
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.32 * len(vids) + 1)))
    ax.barh(vids, [per_video[v]["images"] for v in vids],
            color=[SPLIT_COLORS.get(per_video[v]["split"], "#888") for v in vids])
    ax.invert_yaxis()
    ax.set_xlabel("imagens")
    ax.set_title("Imagens por vídeo (cor = split)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SPLIT_COLORS.values()]
    ax.legend(handles, SPLIT_COLORS.keys(), frameon=False, loc="lower right")
    return fig


def _fig_sizes(classes, sizes):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    data = [sizes[c] if sizes[c] else [np.nan] for c in range(len(classes))]
    ax.boxplot(data, vert=False, tick_labels=classes, showfliers=True,
               flierprops={"markersize": 2, "alpha": 0.4})
    ax.set_xscale("log")
    ticks = [t for t in (8, 16, 32, 64, 128, 256, 512, 1024) if t <= max(1024, 2 * max([max(v) for v in sizes.values() if v] + [1]))]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    for v, t in ((COCO_SMALL, "pequeno | médio"), (COCO_MED, "médio | grande")):
        ax.axvline(v, color="#888", ls="--", lw=1)
        ax.text(v, ax.get_ylim()[1], f" {t}", fontsize=7, va="bottom")
    ax.set_xlabel("tamanho da caixa: √(largura×altura) em pixels da imagem original")
    ax.set_title("Tamanho dos objetos por classe", pad=14)
    return fig


def _fig_heatmaps(classes, centers):
    fig, axes = plt.subplots(1, len(classes), figsize=(3.2 * len(classes), 2.6), squeeze=False)
    for c, ax in enumerate(axes[0]):
        pts = np.array(centers[c]) if centers[c] else np.zeros((0, 2))
        h, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=(9, 16), range=[[0, 1], [0, 1]]) \
            if len(pts) else (np.zeros((9, 16)), 0, 0)
        ax.imshow(h, cmap="magma", aspect="auto", extent=[0, 1, 1, 0])
        ax.set_title(classes[c], fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(False)
    fig.suptitle("Posição dos centros das caixas na imagem (claro = mais instâncias)", fontsize=10, y=1.04)
    return fig


def _fig_cooc(classes, cooc):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cooc, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cooc[i, j]), ha="center", va="center",
                    color="white" if cooc[i, j] > cooc.max() * 0.6 else "black", fontsize=9)
    ax.grid(False)
    ax.set_title("Imagens em que duas classes aparecem juntas\n(diagonal = imagens com a classe)", fontsize=9)
    return fig


def _fig_boxes_per_image(counts, splits):
    mx = max([k for s in splits for k in counts[s]] + [1])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bottom = np.zeros(mx + 1)
    for s in splits:
        vals = np.array([counts[s].get(k, 0) for k in range(mx + 1)], dtype=float)
        ax.bar(range(mx + 1), vals, bottom=bottom, label=s, color=SPLIT_COLORS.get(s, "#888"))
        bottom += vals
    ax.set_xlabel("caixas na imagem (0 = fundo)")
    ax.set_ylabel("imagens")
    ax.set_title("Quantidade de objetos por imagem")
    ax.legend(frameon=False)
    return fig


def _fig_quality(imgs, splits):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, key, title in ((axes[0], "blur", "Nitidez (var. Laplaciano)"),
                           (axes[1], "brightness", "Brilho médio (0-255)")):
        data = [[i[key] for i in imgs if i["split"] == s] or [np.nan] for s in splits]
        bp = ax.boxplot(data, tick_labels=splits, patch_artist=True, showfliers=False)
        for patch, s in zip(bp["boxes"], splits):
            patch.set_facecolor(SPLIT_COLORS.get(s, "#888"))
            patch.set_alpha(0.6)
        ax.set_title(title, fontsize=9)
    return fig


def _contact_sheet(cfg, frames, n=12, tw=200):
    fdir = path_of(cfg, "frames") / "selected"
    tiles = []
    for f in frames[:n]:
        img = imread(fdir / f"{f}.jpg")
        if img is None:
            continue
        h, w = img.shape[:2]
        tiles.append(cv2.resize(img, (tw, round(h * tw / w)), interpolation=cv2.INTER_AREA))
    if not tiles:
        return ""
    th = max(t.shape[0] for t in tiles)
    cols = 4
    rows_ = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    canvas = []
    for r in rows_:
        r = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in r]
        while len(r) < cols:
            r.append(np.full((th, tw + 4, 3), 255, np.uint8))
        canvas.append(np.hstack(r))
    ok, buf = cv2.imencode(".jpg", np.vstack(canvas), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ----------------------------------------------------------------------------
# Relatório
# ----------------------------------------------------------------------------
CSS = """
:root{--bg:#fbfbf9;--fg:#1d2330;--mut:#667085;--line:#dcdfe4;--card:#fff;--warn:#b54708;--warnbg:#fff4e5;--ok:#1b7a43;--okbg:#e9f6ee;--err:#b42318;--errbg:#fdecea}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e8eaee;--mut:#9aa3b2;--line:#2b313b;--card:#1b1f26;--warn:#f5a54a;--warnbg:#2c2213;--ok:#5fcf8b;--okbg:#14261c;--err:#f28b82;--errbg:#2c1816}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 "Segoe UI",system-ui,-apple-system,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:1.7rem;margin:0 0 4px}
h2{font-size:1.15rem;margin:36px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
p.sub{color:var(--mut);margin:0 0 20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 14px}
.kpi b{display:block;font-size:1.5rem;font-variant-numeric:tabular-nums}
.kpi span{color:var(--mut);font-size:.85rem}
.al{margin:6px 0;padding:8px 12px;border-left:4px solid;border-radius:3px}
.al.warn{border-color:var(--warn);background:var(--warnbg)}
.al.ok{border-color:var(--ok);background:var(--okbg)}
.al.err{border-color:var(--err);background:var(--errbg)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.92rem;margin:8px 0}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600}
.scroll{overflow-x:auto}
img{max-width:100%;height:auto;display:block;margin:10px 0;background:#fff;border-radius:4px}
.note{color:var(--mut);font-size:.88rem}
"""


def _table(head, rows):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def generate(cfg: dict, excluded: list[str] | None = None, issues: list[dict] | None = None) -> Path:
    classes = cfg["classes"]
    splits = list(cfg["split"]["ratios"])
    rdir = path_of(cfg, "reports")
    figdir = rdir / "figuras"
    min_inst = cfg["targets"]["min_instances_per_class"]
    bg_target = cfg["targets"]["background_fraction"]
    imgs = _collect(cfg)
    nC = len(classes)
    if issues is None and (rdir / "problemas_anotacao.csv").exists():
        with open(rdir / "problemas_anotacao.csv", newline="", encoding="utf-8") as f:
            issues = list(csv.DictReader(f))

    inst = {s: Counter() for s in splits}
    img_with = {s: Counter() for s in splits}
    n_img, n_bg = Counter(), Counter()
    per_video = defaultdict(lambda: {"images": 0, "inst": Counter(), "split": ""})
    sizes = defaultdict(list)
    centers = defaultdict(list)
    cooc = np.zeros((nC, nC))
    bpi = {s: Counter() for s in splits}

    for im in imgs:
        s = im["split"]
        n_img[s] += 1
        bpi[s][len(im["boxes"])] += 1
        if not im["boxes"]:
            n_bg[s] += 1
        pv = per_video[im["video"]]
        pv["images"] += 1
        pv["split"] = s
        present = set()
        for c, x, y, w, h in im["boxes"]:
            inst[s][c] += 1
            pv["inst"][c] += 1
            sizes[c].append(float(np.sqrt(w * im["W"] * h * im["H"])))
            centers[c].append((x, y))
            present.add(c)
        for c in present:
            img_with[s][c] += 1
            for d in present:
                cooc[c, d] += 1

    tot_img = len(imgs)
    tot_bg = sum(n_bg.values())
    tot_inst_c = [sum(inst[s][c] for s in splits) for c in range(nC)]
    bg_frac = tot_bg / tot_img if tot_img else 0.0

    # ---- alertas
    al = []
    for c in range(nC):
        if tot_inst_c[c] < min_inst:
            al.append(("warn", f"<b>{classes[c]}</b>: {tot_inst_c[c]} instâncias, abaixo da meta de {min_inst}."))
        for s in splits:
            if inst[s][c] == 0 and tot_inst_c[c] > 0:
                al.append(("err", f"<b>{classes[c]}</b> não aparece no split <b>{s}</b>. "
                                  "A métrica desse split não cobre a classe; ajuste a seed ou anote mais vídeos."))
    for c in range(nC):
        if tot_inst_c[c] and "train" in inst and inst["train"][c] / tot_inst_c[c] < 0.5:
            al.append(("warn", f"<b>{classes[c]}</b>: só {inst['train'][c] / tot_inst_c[c]:.0%} das instâncias ficaram no treino "
                               f"(esperado ~{cfg['split']['ratios']['train']:.0%}). Com poucos vídeos o split por vídeo fica desigual; "
                               "tente outra seed em [split] ou anote mais vídeos."))
    if bg_frac < bg_target:
        al.append(("warn", f"Imagens de fundo: {bg_frac:.1%} (meta ≥ {bg_target:.0%}). "
                           "Deixe frames sem objeto sem anotar no CVAT, ou grave trechos só com cenário."))
    big = max(tot_inst_c) if tot_inst_c else 0
    small_c = min(tot_inst_c) if tot_inst_c else 0
    if small_c and big / small_c > 3:
        al.append(("warn", f"Desbalanceamento: a classe mais frequente tem {big / small_c:.1f}× mais instâncias que a menos frequente."))
    small_share = {c: np.mean([v < COCO_SMALL for v in sizes[c]]) if sizes[c] else 0 for c in range(nC)}
    for c, sh in small_share.items():
        if sh > 0.3:
            al.append(("warn", f"<b>{classes[c]}</b>: {sh:.0%} das caixas são pequenas (&lt; {COCO_SMALL}px). "
                               "Considere imgsz maior no treino."))
    for s in splits:
        if n_img[s] and n_img[s] / tot_img < 0.05:
            al.append(("warn", f"Split <b>{s}</b> tem só {n_img[s]} imagens ({n_img[s] / tot_img:.0%})."))
    n_err = sum(1 for i in (issues or []) if i["nivel"] == "erro")
    n_warn = sum(1 for i in (issues or []) if i["nivel"] == "aviso")
    if n_err or n_warn:
        al.append(("warn" if not n_err else "err",
                   f"Validação dos labels: {n_err} erro(s) e {n_warn} aviso(s) (veja problemas_anotacao.csv)."))
    if excluded:
        al.append(("warn", f"{len(excluded)} imagem(ns) de lotes ainda não exportados do CVAT ficaram de fora."))
    if not al:
        al.append(("ok", "Nenhum alerta: metas de instâncias e de fundo atendidas, todas as classes presentes em todos os splits."))

    # ---- figuras
    figs = {
        "instancias_por_classe": _b64_png(_fig_instances(classes, splits, inst, min_inst), figdir / "instancias_por_classe.png"),
        "imagens_por_video": _b64_png(_fig_videos(per_video), figdir / "imagens_por_video.png"),
        "tamanho_objetos": _b64_png(_fig_sizes(classes, sizes), figdir / "tamanho_objetos.png"),
        "mapa_calor": _b64_png(_fig_heatmaps(classes, centers), figdir / "mapa_calor_posicao.png"),
        "coocorrencia": _b64_png(_fig_cooc(classes, cooc), figdir / "coocorrencia.png"),
        "objetos_por_imagem": _b64_png(_fig_boxes_per_image(bpi, splits), figdir / "objetos_por_imagem.png"),
        "qualidade": _b64_png(_fig_quality(imgs, splits), figdir / "nitidez_brilho.png"),
    }
    bg_frames = [i["frame"] for i in imgs if not i["boxes"]]
    sheet = _contact_sheet(cfg, bg_frames)

    # ---- tabelas
    t_split = _table(["split", "vídeos", "imagens", "fundo", "instâncias"],
                     [[s, sum(1 for v in per_video.values() if v["split"] == s), n_img[s], n_bg[s],
                       sum(inst[s].values())] for s in splits]
                     + [["total", len(per_video), tot_img, tot_bg, sum(tot_inst_c)]])
    t_class = _table(["classe"] + [f"{s} (inst / img)" for s in splits] + ["total inst.", "% do total"],
                     [[classes[c]] + [f"{inst[s][c]} / {img_with[s][c]}" for s in splits]
                      + [tot_inst_c[c], f"{tot_inst_c[c] / max(sum(tot_inst_c), 1):.1%}"] for c in range(nC)])
    size_rows = []
    for c in range(nC):
        v = np.array(sizes[c]) if sizes[c] else np.array([])
        if len(v):
            size_rows.append([classes[c], int((v < COCO_SMALL).sum()), int(((v >= COCO_SMALL) & (v < COCO_MED)).sum()),
                              int((v >= COCO_MED).sum()), f"{np.median(v):.0f}"])
        else:
            size_rows.append([classes[c], 0, 0, 0, "-"])
    t_size = _table(["classe", f"pequenas (<{COCO_SMALL}px)", f"médias ({COCO_SMALL}-{COCO_MED}px)",
                     f"grandes (≥{COCO_MED}px)", "mediana (px)"], size_rows)
    vids = sorted(per_video, key=lambda v: (per_video[v]["split"], v))
    t_vid = _table(["vídeo", "split", "imagens"] + classes,
                   [[v, per_video[v]["split"], per_video[v]["images"]] + [per_video[v]["inst"][c] for c in range(nC)]
                    for v in vids])

    # ---- CSVs / JSON
    with open(rdir / "stats_por_classe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["classe", "split", "instancias", "imagens_com_a_classe"])
        for c in range(nC):
            for s in splits:
                w.writerow([classes[c], s, inst[s][c], img_with[s][c]])
    with open(rdir / "stats_por_video.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "split", "imagens"] + classes)
        for v in vids:
            w.writerow([v, per_video[v]["split"], per_video[v]["images"]] + [per_video[v]["inst"][c] for c in range(nC)])
    (rdir / "resumo.json").write_text(json.dumps({
        "imagens": tot_img, "fundo": tot_bg, "fracao_fundo": round(bg_frac, 4),
        "instancias_por_classe": dict(zip(classes, tot_inst_c)),
        "imagens_por_split": dict(n_img),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- HTML
    alerts_html = "".join(f'<div class="al {k}">{m}</div>' for k, m in al)
    kpis = [
        (tot_img, "imagens anotadas"), (sum(tot_inst_c), "instâncias"), (len(per_video), "vídeos"),
        (f"{bg_frac:.1%}", f"fundo (meta ≥ {bg_target:.0%})"),
    ]
    kpi_html = "".join(f'<div class="kpi"><b>{v}</b><span>{html.escape(l)}</span></div>' for v, l in kpis)
    sheet_html = (f'<img alt="Amostra de imagens de fundo" src="{sheet}">'
                  f'<p class="note">Confira: estas imagens estão sem nenhuma caixa. Se aparecer objeto aqui, '
                  f'faltou anotar.</p>') if sheet else '<p class="note">Nenhuma imagem de fundo.</p>'

    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relatório do dataset</title><style>{CSS}</style></head><body><main>
<h1>Composição do dataset</h1>
<p class="sub">Classes: {html.escape(", ".join(classes))}. Split feito por vídeo (seed {cfg['split']['seed']}).</p>
<div class="kpis">{kpi_html}</div>
<h2>Alertas</h2>{alerts_html}
<h2>Splits</h2>{t_split}
<h2>Classes por split</h2>{t_class}
<p class="note">"inst" = caixas anotadas; "img" = imagens que contêm ao menos uma caixa da classe.</p>
<img alt="Instâncias por classe e split" src="{figs['instancias_por_classe']}">
<h2>Vídeos</h2><img alt="Imagens por vídeo" src="{figs['imagens_por_video']}">{t_vid}
<h2>Tamanho dos objetos</h2><img alt="Tamanho dos objetos" src="{figs['tamanho_objetos']}">{t_size}
<h2>Onde os objetos aparecem</h2><img alt="Mapa de calor" src="{figs['mapa_calor']}">
<p class="note">Relevante para o servo-visual: se os objetos só aparecem no centro, o modelo pode
falhar quando o alvo estiver na borda do quadro durante o rastreamento.</p>
<h2>Coocorrência e densidade</h2>
<img alt="Coocorrência" src="{figs['coocorrencia']}" style="max-width:480px">
<img alt="Objetos por imagem" src="{figs['objetos_por_imagem']}">
<h2>Qualidade das imagens</h2><img alt="Nitidez e brilho" src="{figs['qualidade']}">
<h2>Imagens de fundo</h2>{sheet_html}
<p class="note">Figuras em alta resolução: reports/figuras/. Tabelas: stats_por_classe.csv e stats_por_video.csv.</p>
</main></body></html>"""
    out = rdir / "dataset_report.html"
    out.write_text(page, encoding="utf-8")
    print(f"\nRelatório: {out}")
    for _, m in al:
        print("  - " + m.replace("<b>", "").replace("</b>", "").replace("&lt;", "<"))
    return out
