#!/usr/bin/env python3
"""Build CNN embeddings for images using a pretrained model (ResNet18 by default).

Saves embeddings to a joblib file with keys: 'filenames' and 'embeddings' (numpy array).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterable

try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
    from PIL import Image
    import numpy as np
    import joblib
except Exception as e:  # pragma: no cover - environment dependent
    print("Missing runtime dependency for embeddings script. Please install: torch torchvision pillow joblib numpy")
    raise


def list_image_files(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def load_model(name: str, device: str):
    name = name.lower()
    if name == 'resnet18':
        m = models.resnet18(pretrained=True)
        # Remove final fc to get embeddings
        backbone = torch.nn.Sequential(*list(m.children())[:-1])
        out_dim = m.fc.in_features
    elif name == 'resnet50':
        m = models.resnet50(pretrained=True)
        # Remove final fc to get embeddings
        backbone = torch.nn.Sequential(*list(m.children())[:-1])
        out_dim = m.fc.in_features
    else:
        raise ValueError(f"Unsupported model: {name}")
    backbone.to(device)
    backbone.eval()
    return backbone, out_dim


def make_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def batch_iterable(it: Iterable, batch_size: int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_embeddings(image_dir: str, model_name: str, batch_size: int, device: str, limit: int | None = None):
    files = list_image_files(image_dir)
    if limit:
        files = files[:limit]
    transform = make_transform()
    model, out_dim = load_model(model_name, device)
    embeddings = []
    filenames = []

    for batch in batch_iterable(files, batch_size):
        imgs = []
        for fname in batch:
            try:
                img = Image.open(os.path.join(image_dir, fname)).convert('RGB')
                imgs.append(transform(img))
            except Exception as e:
                logging.debug("Skipping %s: %s", fname, e)
                imgs.append(None)
        tensors = [t for t in imgs if t is not None]
        if not tensors:
            # nothing in this batch
            for i, fname in enumerate(batch):
                if imgs[i] is None:
                    continue
            continue
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            out = model(batch_tensor)
            # resnet returns (N, C, 1, 1)
            out = out.reshape(out.shape[0], -1).cpu().numpy()
        # map outputs back to filenames, skipping any that failed load
        ti = 0
        for i, fname in enumerate(batch):
            if imgs[i] is None:
                continue
            embeddings.append(out[ti])
            filenames.append(fname)
            ti += 1

    if not embeddings:
        raise RuntimeError("No embeddings computed (no images or all failed)")

    return np.vstack(embeddings), filenames


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images (fast mode)')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--output', default=None, help='Path to write embeddings joblib')
    args = parser.parse_args(argv or sys.argv[1:])

    out_path = args.output or os.path.join('.cache', f'embeddings_{args.model}.joblib')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'

    logging.info('Building embeddings model=%s device=%s', args.model, device)
    emb, fnames = build_embeddings(args.image_dir, args.model, args.batch_size, device, limit=args.limit)
    logging.info('Built embeddings shape=%s for %d files', emb.shape, len(fnames))
    joblib.dump({'filenames': fnames, 'embeddings': emb}, out_path)
    logging.info('Saved embeddings to %s', out_path)


if __name__ == '__main__':
    main()
