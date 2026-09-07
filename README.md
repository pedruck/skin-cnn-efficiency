# Comparação de eficiência de CNNs para classificação de lesões de pele (ISIC 2019)

Estudo comparativo do compromisso **desempenho × custo computacional** de arquiteturas
CNN pré-treinadas (ImageNet) aplicadas por *transfer learning* à classificação
multiclasse de imagens dermatoscópicas do **ISIC 2019** (8 classes).

Modelos comparados (padrão): **ResNet-50, ResNet-18, MobileNetV2, MobileNetV3-Large**
— duas famílias (ResNet e MobileNet), duas escalas cada, para comparação intra e entre famílias.
Adicionar/trocar modelos = editar a lista `models:` em [`config.yaml`](config.yaml)
(arquiteturas disponíveis na fábrica: `src/models.py`).

## O que é medido

| Dimensão | Métricas |
|---|---|
| **Desempenho preditivo** | acurácia, acurácia balanceada, **macro-F1** (principal), F1 ponderado, κ de Cohen, AUC macro (OvR), recall de MEL (melanoma) |
| **Custo de hardware** | nº de parâmetros, tamanho do modelo (MB), GMACs (≈ GFLOPs/2), latência CPU e GPU (média ± desvio, p95), throughput, VRAM de pico no treino, tempo/energia de treino |
| **Custo-benefício** | macro-F1 por GMAC e por milhão de parâmetros |

Métricas sensíveis a desbalanceamento porque o ISIC 2019 é fortemente desbalanceado
(NV ≈ 12,9 k imagens vs. DF ≈ 240). O `CrossEntropyLoss` é ponderado pela frequência
de classe no treino e o split é **estratificado 70/15/15**, persistido em
`results/splits.csv` e reutilizado por todos os modelos.

## Estrutura

```
config.yaml            toda a configuração do estudo (seed, split, hiperparâmetros, modelos)
run.ipynb              notebook driver para o Google Colab
src/
  data.py              Dataset ISIC 2019, split estratificado, DataLoaders, pesos de classe
  models.py            fábrica build_model(nome, num_classes)
  engine.py            train_one_epoch / evaluate
  metrics.py           métricas preditivas (sklearn)
  efficiency.py        parâmetros, MACs, latência (b=1 e lote), VRAM, energia
  env.py               captura de ambiente -> environment.json
  train.py             CLI: treina 1 modelo, resumível (checkpoint por época)
  benchmark.py         CLI: avalia no teste + perfila eficiência -> results/*.csv, *.npz
  plots.py             CLI: results/ -> figs/*.pdf (+ .png 300 dpi)
scripts/run_all.sh     pipeline completo
results/  checkpoints/  figs/   (gerados; fora do git)
paper/                 template LaTeX IEEE + refs.bib + figuras do artigo
```

## Como rodar no Google Colab

1. Abra [`run.ipynb`](run.ipynb) no Colab (`File ▸ Open notebook ▸ GitHub`).
2. `Runtime ▸ Change runtime type ▸ GPU`.
3. Rode as células na ordem. Elas: clonam o repo, instalam dependências, montam o
   Google Drive (para persistir `results/`, `checkpoints/`, `figs/`), baixam o
   dataset via `kagglehub` e executam treino → benchmark → figuras.

Treine **um modelo por célula/sessão**. O treino é resumível: se a sessão do Colab
cair, basta re-executar a mesma célula — ele continua da última época salva no Drive.

## Como rodar localmente

```bash
pip install -r requirements.txt
# dataset: informe --data OU deixe o kagglehub baixar (precisa de ~/.kaggle/kaggle.json)
bash scripts/run_all.sh /caminho/para/isic-2019
```

Ou passo a passo:

```bash
python -m src.train --model resnet50 --data /caminho/isic-2019 --epochs 20
python -m src.train --model mobilenet_v3_large --data /caminho/isic-2019
python -m src.benchmark --data /caminho/isic-2019
python -m src.plots --data /caminho/isic-2019
```

## Saídas → artigo

| Arquivo gerado | Vira no artigo |
|---|---|
| `results/splits.csv` | Tabela I (composição do dataset) |
| `results/environment.json` | Tabela de ambiente de hardware/software |
| `results/performance.csv` | Tabela de desempenho preditivo |
| `results/efficiency.csv` | Tabela de eficiência |
| `results/summary.csv` | Tabela de custo-benefício |
| `results/per_class_<modelo>.csv` | Métricas por classe |
| `results/test_predictions_<modelo>.npz` | regenera matriz de confusão, PR, F1/classe |
| `figs/fig1_samples` … `fig8_pr_curves` | Figuras 1–8 |

Figura-chave: `fig4_tradeoff` (macro-F1 × GMACs / latência / parâmetros).

## Reprodutibilidade

`seed` e todos os hiperparâmetros ficam em `config.yaml`. Para resultados
bit-a-bit reprodutíveis (≈10–20 % mais lento), defina `deterministic: true`.
O ideal é repetir com ≥3 seeds e reportar média ± desvio nas métricas preditivas.
