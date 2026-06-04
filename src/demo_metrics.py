from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def make_scores(rows=500, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.choice([0, 1], rows, p=[0.65, 0.35])
    score = np.where(y == 1, rng.normal(0.72, 0.12, rows), rng.normal(0.28, 0.12, rows))
    score = np.clip(score, 0, 1)
    return pd.DataFrame({'sample_id': range(1, rows + 1), 'label': y, 'score': score.round(4)})


def metric_row(df, threshold):
    pred = (df['score'] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(df['label'], pred).ravel()
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(df['label'], pred),
        'precision': precision_score(df['label'], pred, zero_division=0),
        'recall': recall_score(df['label'], pred, zero_division=0),
        'f1': f1_score(df['label'], pred, zero_division=0),
        'roc_auc': roc_auc_score(df['label'], df['score']),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='reports')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = make_scores()
    df.to_csv(out / 'demo_scores.csv', index=False)
    pd.DataFrame([metric_row(df, args.threshold)]).round(4).to_csv(out / 'metrics.csv', index=False)
    print(out)


if __name__ == '__main__':
    main()
