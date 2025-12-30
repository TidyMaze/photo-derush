#!/usr/bin/env python3
"""Benchmark CNN training on CPU vs MPS (Metal) GPU."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import RatingsTagsRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("benchmark")


class ImageDataset(Dataset):
    """PyTorch dataset for images."""
    
    def __init__(self, image_paths: list[str], labels: list[int], transform=None, image_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_size = image_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = self.labels[idx]
            return img, label
        except Exception as e:
            log.warning(f"Error loading {self.image_paths[idx]}: {e}")
            img = Image.new("RGB", self.image_size, (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]


class SimpleCNN(nn.Module):
    """Simple CNN architecture."""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def build_image_dataset(image_dir: str, repo: RatingsTagsRepository, limit: int = 50) -> tuple[list[str], list[int]]:
    """Build small dataset for benchmarking."""
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return [], []
    
    all_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    image_paths = []
    labels = []
    
    for fname in all_files:
        if len(image_paths) >= limit:
            break
        state = repo.get_state(fname)
        if state not in ("keep", "trash"):
            continue
        source = repo.get_label_source(fname)
        if source != "manual":
            continue
        
        image_paths.append(os.path.join(image_dir, fname))
        labels.append(1 if state == "keep" else 0)
    
    log.info(f"Using {len(image_paths)} images for benchmark: {sum(labels)} keep, {len(labels) - sum(labels)} trash")
    return image_paths, labels


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def benchmark_device(
    image_paths: list[str],
    labels: list[int],
    train_indices: np.ndarray,
    device_name: str,
    batch_size: int = 16,
    epochs: int = 5,
    warmup_epochs: int = 1,
) -> float:
    """Benchmark training on a specific device."""
    device = torch.device(device_name)
    
    # Create dataset
    train_dataset = ImageDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Model
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Warmup
    log.info(f"  Warmup ({warmup_epochs} epoch)...")
    for _ in range(warmup_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # Synchronize if using GPU
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    log.info(f"  Benchmarking ({epochs} epochs)...")
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # Synchronize again
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start_time
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs MPS")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--limit", type=int, default=50, help="Number of images to use")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to benchmark")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("CNN DEVICE BENCHMARK")
    log.info("="*80)
    
    # Build small dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    image_paths, labels = build_image_dataset(image_dir, repo, limit=args.limit)
    
    if len(labels) < 10:
        log.error(f"Insufficient data: {len(labels)} samples")
        return 1
    
    # Split
    indices = np.arange(len(image_paths))
    train_indices, _ = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    
    log.info(f"\nBenchmarking with {len(train_indices)} training samples")
    log.info(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    
    # Check available devices
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    
    log.info(f"\nAvailable devices:")
    log.info(f"  CPU: Always available")
    log.info(f"  MPS (Metal): {mps_available}")
    log.info(f"  CUDA: {cuda_available}")
    
    results = {}
    
    # Benchmark CPU
    log.info("\n" + "="*80)
    log.info("BENCHMARKING CPU")
    log.info("="*80)
    try:
        cpu_time = benchmark_device(
            image_paths, labels, train_indices,
            device_name="cpu",
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
        results["cpu"] = cpu_time
        log.info(f"CPU time: {cpu_time:.2f}s ({cpu_time/args.epochs:.3f}s per epoch)")
    except Exception as e:
        log.error(f"CPU benchmark failed: {e}")
        return 1
    
    # Benchmark MPS if available
    if mps_available:
        log.info("\n" + "="*80)
        log.info("BENCHMARKING MPS (Metal GPU)")
        log.info("="*80)
        try:
            mps_time = benchmark_device(
                image_paths, labels, train_indices,
                device_name="mps",
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
            results["mps"] = mps_time
            log.info(f"MPS time: {mps_time:.2f}s ({mps_time/args.epochs:.3f}s per epoch)")
        except Exception as e:
            log.error(f"MPS benchmark failed: {e}")
            results["mps"] = None
    
    # Results summary
    log.info("\n" + "="*80)
    log.info("RESULTS SUMMARY")
    log.info("="*80)
    
    cpu_time = results.get("cpu")
    mps_time = results.get("mps")
    
    if cpu_time:
        log.info(f"\nCPU:   {cpu_time:.2f}s ({cpu_time/args.epochs:.3f}s/epoch)")
    
    if mps_time:
        log.info(f"MPS:   {mps_time:.2f}s ({mps_time/args.epochs:.3f}s/epoch)")
        if cpu_time:
            speedup = cpu_time / mps_time
            log.info(f"\nSpeedup: MPS is {speedup:.2f}x faster than CPU")
            if speedup > 1.0:
                log.info(f"✅ Use MPS (Metal GPU) for training")
            else:
                log.info(f"⚠️  CPU is faster (MPS overhead may be significant for small batches)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

