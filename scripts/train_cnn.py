#!/usr/bin/env python3
"""Train CNN directly on images (no handcrafted features).

Uses PyTorch to train a convolutional neural network on raw images.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import RatingsTagsRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("train_cnn")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not available, trying TensorFlow/Keras...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
        log.error("Neither PyTorch nor TensorFlow available. Install one of them.")


class ImageDataset(Dataset):
    """PyTorch dataset for images."""
    
    def __init__(self, image_paths: list[str], labels: list[int], transform=None, image_size=(224, 224), is_train=False):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        
        if transform is None:
            if is_train:
                # Training: lighter data augmentation for speed
                self.transform = transforms.Compose([
                    transforms.Resize((240, 240)),  # Smaller resize
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),  # Lighter jitter
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
                ])
            else:
                # Validation/test: no augmentation
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform
    
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
            # Return black image and label 0 on error
            img = Image.new("RGB", self.image_size, (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]


class TransferLearningModel(nn.Module):
    """Transfer learning model using pre-trained ResNet18."""
    
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(TransferLearningModel, self).__init__()
        # Load pre-trained ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze backbone layers initially
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class SimpleCNN(nn.Module):
    """Small CNN architecture - reduced size for smaller datasets."""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Convolutional layers - smaller channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers - smaller
        # After 3 pooling layers: 224 / 8 = 28
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 224->112
        x = self.pool(self.relu(self.conv2(x)))  # 112->56
        x = self.pool(self.relu(self.conv3(x)))  # 56->28
        
        x = x.view(x.size(0), -1)  # Flatten: 64*28*28 = 50,176
        x = self.dropout(self.relu(self.fc1(x)))  # 50,176 -> 128
        x = self.fc2(x)  # 128 -> 2
        return x


def build_image_dataset(image_dir: str, repo: RatingsTagsRepository) -> tuple[list[str], list[int]]:
    """Build dataset of image paths and labels."""
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return [], []
    
    all_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    image_paths = []
    labels = []
    
    for fname in all_files:
        state = repo.get_state(fname)
        if state not in ("keep", "trash"):
            continue
        source = repo.get_label_source(fname)
        if source != "manual":
            continue
        
        image_paths.append(os.path.join(image_dir, fname))
        labels.append(1 if state == "keep" else 0)
    
    log.info(f"Found {len(image_paths)} labeled images: {sum(labels)} keep, {len(labels) - sum(labels)} trash")
    return image_paths, labels


def train_cnn_pytorch(
    image_paths: list[str],
    labels: list[int],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.001,
    device: str = "auto",
    use_transfer: bool = True,
    fine_tune: bool = False,
) -> tuple[float, float]:
    """Train CNN using PyTorch with transfer learning and data augmentation."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    # Auto-detect best device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("Using Apple Metal (MPS) GPU")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            log.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            log.info("Using CPU (no GPU available)")
    elif device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("Using Apple Metal (MPS) GPU")
        else:
            log.warning("MPS requested but not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        if device == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS requested but not available, falling back to CPU")
            device = torch.device("cpu")
        elif device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(device)
    
    log.info(f"Using device: {device}")
    
    # Create datasets with augmentation for training
    train_dataset = ImageDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        is_train=True,
    )
    test_dataset = ImageDataset(
        [image_paths[i] for i in test_indices],
        [labels[i] for i in test_indices],
        is_train=False,
    )
    
    # Use larger batch size for validation (faster evaluation)
    val_batch_size = min(batch_size * 2, 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Use transfer learning model or simple CNN
    if use_transfer:
        model = TransferLearningModel(num_classes=2, freeze_backbone=not fine_tune).to(device)
        log.info("Using transfer learning with pre-trained ResNet18")
        if fine_tune:
            log.info("Fine-tuning entire model (backbone unfrozen)")
        else:
            log.info("Training classifier head only (backbone frozen)")
    else:
        model = SimpleCNN(num_classes=2).to(device)
        log.info("Using simple CNN architecture")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Early stopping
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    # Training
    log.info("Starting training with data augmentation...")
    log.info(f"Training for up to {epochs} epochs (early stopping patience: {patience})")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        train_acc = correct / total if total > 0 else 0.0
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase (every epoch, but faster with larger batch)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log every epoch
        log.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            log.info(f"  ✓ New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"  Early stopping triggered (no improvement for {patience} epochs)")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log.info(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
    
    # Final evaluation
    log.info("Evaluating best model on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return acc, f1


def train_cnn_tensorflow(
    image_paths: list[str],
    labels: list[int],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.001,
) -> tuple[float, float]:
    """Train CNN using TensorFlow/Keras."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available")
    
    # Load and preprocess images
    def load_image(path):
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))
        return np.array(img) / 255.0
    
    log.info("Loading images...")
    X_train = np.array([load_image(image_paths[i]) for i in train_indices])
    X_test = np.array([load_image(image_paths[i]) for i in test_indices])
    y_train = np.array([labels[i] for i in train_indices])
    y_test = np.array([labels[i] for i in test_indices])
    
    # Model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    
    # Training
    log.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1 if epochs <= 10 else 2,
    )
    
    # Evaluation
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train CNN on raw images")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--output", default=".cache/cnn_results.json", help="Output JSON file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (larger = faster)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default="mps", help="Device (auto/cpu/cuda/mps, default: mps)")
    parser.add_argument("--use-transfer", action="store_true", default=True, help="Use transfer learning (default: True)")
    parser.add_argument("--fine-tune", action="store_true", default=False, help="Fine-tune entire model (unfreeze backbone)")
    args = parser.parse_args()
    
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("CNN TRAINING ON RAW IMAGES")
    log.info("="*80)
    
    # Build dataset
    log.info(f"\nLoading dataset from {image_dir}...")
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    dataset_start = time.perf_counter()
    image_paths, labels = build_image_dataset(image_dir, repo)
    dataset_time = time.perf_counter() - dataset_start
    
    if len(labels) < 20:
        log.error(f"Insufficient labeled data: {len(labels)} samples")
        return 1
    
    log.info(f"Dataset loaded in {dataset_time:.2f}s: {len(labels)} samples")
    log.info(f"  Keep: {sum(labels)}, Trash: {len(labels) - sum(labels)}")
    
    # Split
    indices = np.arange(len(image_paths))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    
    log.info(f"Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    # Train CNN
    log.info("\n" + "="*80)
    log.info("TRAINING CNN")
    log.info("="*80)
    
    train_start = time.perf_counter()
    
    if TORCH_AVAILABLE:
        log.info("Using PyTorch")
        acc, f1 = train_cnn_pytorch(
            image_paths, labels, train_indices, test_indices,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
        )
    elif TF_AVAILABLE:
        log.info("Using TensorFlow/Keras")
        acc, f1 = train_cnn_tensorflow(
            image_paths, labels, train_indices, test_indices,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
    else:
        log.error("Neither PyTorch nor TensorFlow available")
        return 1
    
    train_time = time.perf_counter() - train_start
    
    # Results
    log.info("\n" + "="*80)
    log.info("RESULTS")
    log.info("="*80)
    log.info(f"\nCNN Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    log.info(f"CNN F1: {f1:.4f}")
    log.info(f"Training time: {train_time:.2f}s")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "accuracy": float(acc),
            "f1": float(f1),
            "n_samples": len(labels),
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "training_time": float(train_time),
            "framework": "pytorch" if TORCH_AVAILABLE else "tensorflow",
        }, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

