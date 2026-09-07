"""Laços de treino e avaliacao (agnosticos a arquitetura)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler=None, use_amp: bool = False) -> float:
    model.train()
    running, seen = 0.0, 0
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"

    for images, labels in tqdm(loader, desc="treino", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=autocast_dev, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running += loss.item() * images.size(0)
        seen += images.size(0)

    return running / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Retorna (loss_media, y_true, y_pred, y_prob)."""
    model.eval()
    running, seen = 0.0, 0
    all_true, all_pred, all_prob = [], [], []

    for images, labels in tqdm(loader, desc="avaliacao", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = F.softmax(outputs, dim=1)

        running += loss.item() * images.size(0)
        seen += images.size(0)
        all_true.append(labels.cpu().numpy())
        all_pred.append(probs.argmax(dim=1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return running / max(seen, 1), y_true, y_pred, y_prob
