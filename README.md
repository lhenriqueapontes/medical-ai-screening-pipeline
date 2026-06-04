# Generic ML Screening Pipeline

Projeto demonstrativo de classificacao binaria com scores sinteticos.

## Como executar

```bash
pip install -r requirements.txt
python src/demo_metrics.py --output-dir reports --threshold 0.50
```

## Saidas

- `reports/demo_scores.csv`
- `reports/metrics.csv`

## O que o projeto faz

1. Cria labels e scores ficticios.
2. Aplica um threshold configuravel.
3. Calcula acuracia, precisao, recall, F1, ROC AUC e matriz 2x2.
4. Exporta tabelas para analise.

## Uso

Serve como exemplo generico de avaliacao de um classificador binario. Nao usa imagens reais e nao faz inferencia clinica.

## Licenca

MIT License.
