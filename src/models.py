"""Fabrica de modelos: mesma receita de cabeca de classificacao para todas as arquiteturas.

Adicionar um modelo novo = uma entrada em ``_build`` + ``DISPLAY_NAMES``.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

INPUT_SIZE = (1, 3, 224, 224)

# nome legivel para figuras/tabelas do artigo
DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "resnet18": "ResNet-18",
    "mobilenet_v2": "MobileNetV2",
    "mobilenet_v3_large": "MobileNetV3-L",
    "shufflenet_v2_x1_0": "ShuffleNetV2-1.0x",
}


def _replace_linear(layer: nn.Linear, num_classes: int) -> nn.Linear:
    return nn.Linear(layer.in_features, num_classes)


def _resnet(ctor, weights_enum, num_classes, pretrained):
    m = ctor(weights=weights_enum.DEFAULT if pretrained else None)
    m.fc = _replace_linear(m.fc, num_classes)
    return m


def _build(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    name = name.lower()
    if name == "resnet50":
        return _resnet(models.resnet50, models.ResNet50_Weights, num_classes, pretrained)
    if name == "resnet18":
        return _resnet(models.resnet18, models.ResNet18_Weights, num_classes, pretrained)
    if name == "shufflenet_v2_x1_0":
        return _resnet(models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights,
                       num_classes, pretrained)
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = _replace_linear(m.classifier[1], num_classes)
        return m
    if name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        m.classifier[3] = _replace_linear(m.classifier[3], num_classes)
        return m
    raise ValueError(f"Modelo desconhecido: {name!r}. Conhecidos: {sorted(DISPLAY_NAMES)}")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Cria o modelo com pesos ImageNet e a camada final trocada para ``num_classes``."""
    return _build(name, num_classes, pretrained)


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name.lower(), name)
