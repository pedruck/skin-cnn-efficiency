"""Dataset ISIC 2019, split estratificado 70/15/15 reprodutivel e DataLoaders.

O split e calculado uma unica vez e persistido em ``results/splits.csv``.
Todas as execucoes (todos os modelos) reutilizam esse mesmo arquivo, garantindo
que treino/validacao/teste sao identicos entre os modelos comparados.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Localizacao dos arquivos no diretorio baixado
# --------------------------------------------------------------------------- #
def find_groundtruth_csv(data_dir: str) -> str:
    candidates = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    for c in candidates:
        name = os.path.basename(c).lower()
        if "groundtruth" in name or "ground_truth" in name or "metadata" in name:
            return c
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Nenhum CSV encontrado em {data_dir}")


def build_image_path_map(data_dir: str) -> dict[str, str]:
    paths = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    paths += glob.glob(os.path.join(data_dir, "**", "*.jpeg"), recursive=True)
    mapping = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
    if not mapping:
        raise FileNotFoundError(f"Nenhuma imagem .jpg encontrada em {data_dir}")
    return mapping


# --------------------------------------------------------------------------- #
# DataFrame mestre (imagem -> rotulo inteiro), filtrado ao que existe em disco
# --------------------------------------------------------------------------- #
def build_master_frame(data_dir: str, classes: list[str]) -> pd.DataFrame:
    csv = find_groundtruth_csv(data_dir)
    path_map = build_image_path_map(data_dir)

    raw = pd.read_csv(csv)
    raw["image_base"] = raw["image"].astype(str).str.replace(".jpg", "", regex=False)
    raw = raw[raw["image_base"].isin(path_map)].reset_index(drop=True)

    missing = [c for c in classes if c not in raw.columns]
    if missing:
        raise ValueError(f"Classes ausentes no CSV {csv}: {missing}")

    onehot = raw[classes].to_numpy(dtype="float32")
    keep = onehot.sum(axis=1) > 0  # descarta linhas sem rotulo (UNK)
    raw = raw[keep].reset_index(drop=True)
    onehot = onehot[keep]

    frame = pd.DataFrame({
        "image_base": raw["image_base"].to_numpy(),
        "label": onehot.argmax(axis=1).astype(int),
    })
    frame["class"] = [classes[i] for i in frame["label"]]
    frame["path"] = frame["image_base"].map(path_map)
    return frame


def balance_frame(frame: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Subamostra exatamente ``per_class`` imagens de cada classe.

    Roda ANTES do split, de modo que treino, validacao e teste herdam o
    balanceamento. Deterministico dada a semente.
    """
    counts = frame["class"].value_counts()
    if per_class > counts.min():
        raise ValueError(
            f"balance_per_class={per_class} excede a menor classe: "
            f"{counts.idxmin()} tem {counts.min()} imagens")
    rng = np.random.default_rng(seed)
    keep = [rng.choice(frame.index[frame["class"] == c].to_numpy(),
                       size=per_class, replace=False)
            for c in sorted(frame["class"].unique())]
    return frame.loc[np.concatenate(keep)].sort_index().reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Split estratificado 70/15/15, persistido
# --------------------------------------------------------------------------- #
def load_or_make_splits(frame: pd.DataFrame, val_frac: float, test_frac: float,
                        seed: int, splits_path: str) -> pd.DataFrame:
    if os.path.exists(splits_path):
        splits = pd.read_csv(splits_path)
        print(f"[data] split reutilizado de {splits_path}")
        return splits

    from sklearn.model_selection import train_test_split

    idx = frame.index.to_numpy()
    labels = frame["label"].to_numpy()

    train_idx, temp_idx = train_test_split(
        idx, test_size=val_frac + test_frac, stratify=labels, random_state=seed,
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test, stratify=labels[temp_idx], random_state=seed,
    )

    split = pd.Series("train", index=frame.index)
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"

    splits = frame[["image_base", "label", "class"]].copy()
    splits["split"] = split.to_numpy()
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    splits.to_csv(splits_path, index=False)
    print(f"[data] split criado e salvo em {splits_path}")
    return splits


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(image_size: int, resize_before_crop: int) -> dict:
    """Treino e avaliacao compartilham o mesmo enquadramento.

    Os dois caminhos fazem ``resize_before_crop`` -> recorte de ``image_size``
    (aleatorio no treino, central na avaliacao). Avaliar com a imagem inteira
    reduzida direto a ``image_size`` daria ao modelo um campo de visao mais
    amplo do que ele viu treinando -- um train/eval mismatch que custa alguns
    decimos de macro-F1 de graca.

    O ``Resize`` inicial tambem torna o cache de ``src.prepare_cache``
    transparente: se as imagens ja estao em ``resize_before_crop``, ele nao
    altera nada e o resultado e identico ao do dataset original.
    """
    train_tfm = transforms.Compose([
        transforms.Resize((resize_before_crop, resize_before_crop)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((resize_before_crop, resize_before_crop)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return {"train": train_tfm, "eval": eval_tfm}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ISIC2019Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


# --------------------------------------------------------------------------- #
# API principal
# --------------------------------------------------------------------------- #
def build_dataloaders(cfg: dict, data_dir: str):
    """Retorna (loaders, class_weights, class_names, splits_df)."""
    classes = list(cfg["data"]["classes"])
    frame = build_master_frame(data_dir, classes)

    per_class = cfg["data"].get("balance_per_class")
    if per_class:
        frame = balance_frame(frame, int(per_class), cfg["seed"])
        print(f"[data] balanceado: {per_class} imagens por classe "
              f"({len(frame)} no total, {len(classes)} classes)")

    splits_path = os.path.join(cfg["paths"]["results_dir"], "splits.csv")
    splits = load_or_make_splits(
        frame,
        cfg["data"]["val_fraction"],
        cfg["data"]["test_fraction"],
        cfg["seed"],
        splits_path,
    )

    merged = frame.merge(splits[["image_base", "split"]], on="image_base", how="inner")
    tfm = build_transforms(cfg["data"]["image_size"], cfg["data"]["resize_before_crop"])

    def _loader(split: str, train: bool) -> DataLoader:
        sub = merged[merged["split"] == split].reset_index(drop=True)
        ds = ISIC2019Dataset(sub, tfm["train"] if train else tfm["eval"])
        return DataLoader(
            ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=train,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=train,
        )

    loaders = {
        "train": _loader("train", train=True),
        "val": _loader("val", train=False),
        "test": _loader("test", train=False),
    }

    counts = np.bincount(merged.loc[merged["split"] == "train", "label"], minlength=len(classes))
    weights = counts.sum() / (len(classes) * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    print(f"[data] treino={sum(merged.split == 'train')}  val={sum(merged.split == 'val')}  "
          f"teste={sum(merged.split == 'test')}  classes={len(classes)}")
    return loaders, class_weights, classes, merged
