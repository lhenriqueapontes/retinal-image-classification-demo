from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def features(path):
    img = Image.open(path).convert('RGB').resize((64, 64))
    arr = np.asarray(img) / 255.0
    return np.r_[arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default='data/synthetic_retina/labels.csv')
    parser.add_argument('--output-dir', default='reports')
    args = parser.parse_args()
    df = pd.read_csv(args.labels)
    x = np.vstack([features(p) for p in df['image_path']])
    y = df['label'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'accuracy': accuracy_score(y_test, pred)}]).to_csv(out / 'metrics.csv', index=False)
    print(out)


if __name__ == '__main__':
    main()
