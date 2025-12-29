# Photo Derush

Photo Derush is a desktop-first photo triage tool focused on efficient ML-assisted classification (keep/trash).

## Quick Start

```bash
poetry install
poetry run python app.py --log-file logs/qt-run-latest.log
```

## Key Features

- **Keyboard-first labeling**: K/T keys for keep/trash
- **ML auto-labeling**: XGBoost with calibrated probabilities
- **Batch operations**: Label multiple images at once
- **Object detection**: Visualize detected objects on thumbnails
- **Model training**: Train and evaluate models locally

## Architecture

- **Desktop app**: PySide6 (Qt) - `poetry run python app.py`
- **ML models**: XGBoost models stored in `models/`
- **Data**: SQLite database for labels and metadata
- Run tests: `pytest tests/`

## Documentation

- **Model Documentation**: See `docs/MODEL.md` for best model configuration, metrics guide, and methodology
- **Technical Report**: See `docs/technical_report.md` for comprehensive technical details
- **Studies**: See `docs/studies/` for learning rate studies and AVA dataset experiments
- **Features**: See `docs/features/` for embeddings, PCA, and dataset documentation
- **Improvements**: See `docs/IMPROVEMENTS.md` for future enhancement roadmap

---

## How to Run (KISS)

If you just want to start the app and triage photos with minimal setup, do this:

```bash
# 1) Install dependencies (once)
poetry install

# 2) Run the desktop app
poetry run python app.py

# 3) Optional: write logs to file
poetry run python app.py --log-file logs/manual_run.txt
```

Notes:
- Use the `--log-file` flag to capture output for debugging.
- Default image folder is prompted on first run (or loaded from `~/.photo_app_config.json`).
- To repeat a clean run: close the app and run the same `poetry run` command again.


## 🖥️ Desktop App

Photo Derush is centered on a native desktop application built with PySide6 (Qt). The app provides a fast, keyboard-first workflow for photo triage and local AI-assisted auto-labeling.

### Running the Desktop App

**Prerequisites**
- Python 3.12+
- Poetry

**Quick Start**
```bash
# Install dependencies
poetry install

# Run desktop app
poetry run python app.py

# Or use the helper script
./scripts/run-desktop.sh
```

**With Logging**
```bash
# Enable debug logging to file
poetry run python app.py --log-file debug.log
```

### Desktop App Features

- Native Performance — Fast Qt-based UI
- Offline Mode — Works without network access
- AI Integration — Local XGBoost models with calibrated probabilities
- Thumbnail Grid — Visual photo browsing
- Keyboard Shortcuts — K/T, 1-5 ratings and navigation
- Status Bar — Live image statistics and model status

### Packaging for Distribution

Using PyInstaller is recommended to create portable desktop binaries.

```bash
# Install PyInstaller
poetry add --group dev pyinstaller

# Create standalone executable
poetry run pyinstaller --onefile --windowed \
  --name photo-derush \
  --add-data "src:src" \
  app.py

# Find executable in dist/ folder
ls -la dist/
```

Platform-specific notes: use platform packaging tools (macOS bundles, Windows installers, AppImage) alongside PyInstaller as needed.

---

## ✨ Features

- Fast Labeling — Keyboard shortcuts (K/T, 1-5)
- AI Training — XGBoost models with a feature set for images
- Auto-Labeling — Confidence-filtered auto-label suggestions
- EXIF Display — Camera metadata
- Dark Theme — Professional UI

---

## 🛠 Setup (Quick)

Prerequisites: `Python 3.10+`, `git`.

In 5 minutes:
```bash
# Clone and setup Python
git clone <repo>
cd photo-derush
poetry install

# Run the desktop app
poetry run python app.py
```

Development setup:
- Use Poetry: `poetry install && poetry shell`
- Run tests: `pytest tests/`

---

## 🤖 AI Features

### Training
- Train models on manually labeled photos
- Fast training times for small datasets
- Hyperparameter tuning and cross-validation support

### Prediction
- Auto-label unlabeled images with confidence thresholds
- Bulk predict and review workflows inside the app

---

## 🎨 Technology

- Desktop UI: PySide6 (Qt)
- ML: XGBoost + scikit-learn
- Persistence: local files and model artifacts under `models/`

---

If you'd like, I can open a PR branch with this README change for review, or search other docs to remove remaining web/SAAS references.

```
✅ **Keyboard Navigation** - Arrow keys + shortcuts  


