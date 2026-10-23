"""CLI da pipeline de dataset (vídeo -> CVAT -> YOLO -> Colab).

Uso:
    python pipeline.py extract [--force] [--max-frames N]  # 1) vídeos (novos) -> frames selecionados
    python pipeline.py pack [--batch-size N]               # 2) zips de lotes (novos) para o CVAT
    python pipeline.py build [--allow-issues]  # 3) exports do CVAT -> validação -> split -> relatório
    python pipeline.py report              #    só regenera o relatório
    python pipeline.py sync                # 4) dataset.zip -> Google Drive
    python pipeline.py fetch               # 5) traz resultados do treino do Drive
"""
from __future__ import annotations

import argparse
import sys

if sys.version_info < (3, 11):
    sys.exit("Requer Python 3.11+ (usa tomllib).")

from src import build, colab, extract, pack, report
from src.common import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="caminho alternativo para o .toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--force", action="store_true", help="refaz tudo, apagando frames e manifest anteriores")
    e.add_argument("--max-frames", type=int, help="teto de frames novos nesta execução (padrão: [extract].max_frames)")
    p = sub.add_parser("pack")
    p.add_argument("--batch-size", type=int, help="imagens por lote novo (padrão: [cvat].batch_size)")
    b = sub.add_parser("build")
    b.add_argument("--allow-issues", action="store_true", help="segue mesmo com linhas de label inválidas")
    sub.add_parser("report")
    sub.add_parser("sync")
    sub.add_parser("fetch")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "extract":
        extract.run(cfg, force=args.force, max_frames=args.max_frames)
    elif args.cmd == "pack":
        pack.run(cfg, batch_size=args.batch_size)
    elif args.cmd == "build":
        build.run(cfg, allow_issues=args.allow_issues)
    elif args.cmd == "report":
        report.generate(cfg)
    elif args.cmd == "sync":
        colab.sync(cfg)
    elif args.cmd == "fetch":
        colab.fetch(cfg)


if __name__ == "__main__":
    main()
