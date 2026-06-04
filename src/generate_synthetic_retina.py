from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd


def make_image(label, size=128, seed=0):
    rng = np.random.default_rng(seed)
    img = Image.new('RGB', (size, size), (18, 8, 8))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, size - 8, size - 8), fill=(110, 45, 35))
    draw.ellipse((size * 0.58, size * 0.42, size * 0.70, size * 0.54), fill=(230, 190, 110))
    for _ in range(18):
        x1 = int(size * 0.64)
        y1 = int(size * 0.48)
        x2 = int(rng.integers(15, size - 15))
        y2 = int(rng.integers(15, size - 15))
        draw.line((x1, y1, x2, y2), fill=(150, 55, 45), width=1)
    if label == 1:
        for _ in range(10):
            x = int(rng.integers(25, size - 25))
            y = int(rng.integers(25, size - 25))
            r = int(rng.integers(2, 5))
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(240, 40, 35))
    return img


def generate(output_dir='data/synthetic_retina', n=80, seed=42):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        label = 1 if i >= n // 2 else 0
        image = make_image(label, seed=seed + i)
        filename = f'image_{i:04d}.png'
        image.save(out / filename)
        rows.append({'image_path': str(out / filename), 'label': label})
    pd.DataFrame(rows).to_csv(out / 'labels.csv', index=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/synthetic_retina')
    parser.add_argument('--n', type=int, default=80)
    args = parser.parse_args()
    print(generate(args.output_dir, args.n))


if __name__ == '__main__':
    main()
