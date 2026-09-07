#!/usr/bin/env bash
# Pipeline completo: treino de todos os modelos -> benchmark -> figuras.
# Uso:  bash scripts/run_all.sh /caminho/para/isic2019
# Se o caminho for omitido, cada script baixa o dataset via kagglehub.
set -euo pipefail

DATA_ARG=()
if [[ $# -ge 1 ]]; then
  DATA_ARG=(--data "$1")
fi

MODELS=$(python -c "import yaml;print(' '.join(yaml.safe_load(open('config.yaml'))['models']))")

for m in $MODELS; do
  echo "==================  TREINO: $m  =================="
  python -m src.train --model "$m" "${DATA_ARG[@]}"
done

echo "==================  BENCHMARK  =================="
python -m src.benchmark "${DATA_ARG[@]}"

echo "==================  FIGURAS  =================="
python -m src.plots "${DATA_ARG[@]}"

echo "OK - veja results/ e figs/"
