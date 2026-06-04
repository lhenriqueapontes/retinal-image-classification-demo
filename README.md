# Synthetic Image Classification Demo

Projeto demonstrativo de classificacao de imagens usando figuras sinteticas geradas pelo proprio repositorio.

## Como executar

```bash
pip install -r requirements.txt
python src/generate_synthetic_retina.py --n 80 --output-dir data/synthetic_retina
python src/image_baseline.py --labels data/synthetic_retina/labels.csv --output-dir reports
```

## Saidas

- `data/synthetic_retina/labels.csv`
- `reports/metrics.csv`

## O que o projeto faz

1. Cria imagens sinteticas simples.
2. Gera labels ficticios.
3. Extrai estatisticas basicas de cor.
4. Treina um classificador baseline.
5. Exporta metricas.

## Observacao

Este projeto e apenas uma simulacao tecnica. Nao usa imagens reais.

## Licenca

MIT License.
