"""Pre-redimensiona o dataset em disco, uma unica vez (cache de imagens).

Motivacao
---------
O pipeline de treino ja reduz toda imagem para ``resize_before_crop`` (256) na
primeira transformacao -- o modelo nunca ve mais que isso. Refazer esse resize a
cada epoca, para ~25 mil JPEGs de ~1024x768, faz do DataLoader o gargalo: a GPU
fica ociosa esperando o disco e ``epoch_time_s`` passa a medir o armazenamento
em vez da arquitetura, invalidando a comparacao de custo de treino do artigo.

Cachear e apenas memorizar um calculo deterministico: o que chega ao modelo e o
mesmo. A unica perda e uma segunda compressao JPEG, desprezivel em quality=95.

Uso
---
    python -m src.prepare_cache --data "$DATA" --out /content/isic256
    python -m src.train --model resnet50 --data /content/isic256

E resumivel: re-executar pula o que ja foi convertido, entao uma queda de sessao
no Colab nao custa nada. Verifique antes se vale a pena -- se as imagens de
origem ja forem pequenas, nao ha o que ganhar:

    python -m src.prepare_cache --data "$DATA" --out /tmp/x --inspect
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor

import yaml
from PIL import Image
from tqdm.auto import tqdm

IMAGE_GLOBS = ("*.jpg", "*.jpeg")


def default_size(config_path: str) -> int:
    """Le apenas ``data.resize_before_crop``.

    Deliberadamente nao usa ``src.utils.load_config``: este script so precisa de
    PIL e PyYAML, e importar utils arrastaria torch/numpy junto. Assim ele roda
    em qualquer maquina, inclusive antes de instalar as dependencias pesadas.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return int(yaml.safe_load(f)["data"]["resize_before_crop"])


def find_images(data_dir: str) -> list[str]:
    paths: list[str] = []
    for pattern in IMAGE_GLOBS:
        paths += glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)
    return sorted(paths)


def _resize_one(job: tuple[str, str, int, int]) -> str:
    src, dst, size, quality = job
    if os.path.exists(dst):
        return "skip"
    try:
        with Image.open(src) as im:
            out = im.convert("RGB").resize((size, size), Image.BICUBIC)
        tmp = f"{dst}.tmp"
        out.save(tmp, "JPEG", quality=quality)
        os.replace(tmp, dst)  # atomico: uma queda nunca deixa JPEG pela metade
        return "ok"
    except Exception:
        return "fail"


def _total_mb(paths: list[str]) -> float:
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p)) / 1024**2


def inspect(data_dir: str, sample: int = 8) -> None:
    """Mostra as dimensoes de algumas imagens para decidir se o cache compensa."""
    paths = find_images(data_dir)
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {data_dir}")
    print(f"[inspect] {len(paths)} imagens, {_total_mb(paths):.1f} MB no total")
    for p in random.Random(0).sample(paths, min(sample, len(paths))):
        try:
            with Image.open(p) as im:
                print(f"   {im.size[0]}x{im.size[1]}  {os.path.basename(p)}")
        except Exception as exc:  # arquivo corrompido nao pode derrubar o relatorio
            print(f"   ILEGIVEL  {os.path.basename(p)} ({exc.__class__.__name__})")
    print("[inspect] se as dimensoes ja forem <= 256, o cache nao traz ganho.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="diretorio do ISIC 2019 original")
    ap.add_argument("--out", required=True, help="diretorio de destino do cache")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--size", type=int, default=None,
                    help="padrao: data.resize_before_crop do config.yaml")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--inspect", action="store_true",
                    help="so relata as dimensoes de origem e sai")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.data)
        return

    size = args.size or default_size(args.config)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    srcs = find_images(args.data)
    if not srcs:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {args.data}")

    # nomes achatados: build_image_path_map indexa pelo basename sem extensao
    jobs = [(s, os.path.join(img_dir, os.path.splitext(os.path.basename(s))[0] + ".jpg"),
             size, args.quality) for s in srcs]

    # o CSV de rotulos precisa acompanhar o cache (find_groundtruth_csv o procura la)
    csvs = glob.glob(os.path.join(args.data, "**", "*.csv"), recursive=True)
    for csv in csvs:
        shutil.copy2(csv, os.path.join(args.out, os.path.basename(csv)))
    print(f"[cache] {len(csvs)} CSV(s) copiado(s) para {args.out}")

    workers = args.workers or min(8, os.cpu_count() or 2)
    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(pool.map(_resize_one, jobs, chunksize=32),
                           total=len(jobs), desc=f"cache {size}px"):
            counts[result] += 1

    print(f"[cache] {counts['ok']} convertidas | {counts['skip']} ja existiam | "
          f"{counts['fail']} falharam")
    if counts["fail"]:
        print("[cache] AVISO: imagens que falharam ficam de fora do estudo "
              "(build_master_frame so usa o que existe em disco).")
    print(f"[cache] origem {_total_mb(srcs):.1f} MB  ->  cache "
          f"{_total_mb(glob.glob(os.path.join(img_dir, '*.jpg'))):.1f} MB")
    print(f"\n[cache] pronto. Use com:\n    python -m src.train --model resnet50 --data {args.out}")


if __name__ == "__main__":
    main()
