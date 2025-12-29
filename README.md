# 📸 Photo Derush

> **Tame your photo chaos with AI-powered triage** 🚀

Photo Derush is a desktop-first photo management tool that helps you quickly sort through thousands of photos using **machine learning** and **keyboard-first workflows**. Stop drowning in duplicates, blurry shots, and screenshots—let AI help you decide what to keep and what to trash.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

![Main Interface](screenshots/main-interface.png)

---

## ✨ What Makes This Special?

- **🤖 AI That Actually Works**: CatBoost model with 75%+ accuracy, optimized to minimize false negatives (so you never lose a good photo)
- **⚡ Lightning Fast**: Label photos in seconds with keyboard shortcuts—no mouse required
- **🔒 Privacy First**: Everything runs locally. Your photos never leave your machine
- **🎯 Smart Auto-Labeling**: AI suggests keep/trash labels with confidence scores
- **👁️ Visual Feedback**: See detected objects, EXIF data, and prediction probabilities at a glance
- **🔄 Real-Time Retraining**: The model learns from your decisions and improves as you label

---

## 🎬 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/TidyMaze/photo-derush.git
cd photo-derush

# Install dependencies (requires Poetry)
poetry install

# Run the app
poetry run python app.py
```

**That's it!** The app will prompt you to select your photo directory on first run.

### First Time Using It?

1. **Point it at your photos**: Select a folder with your images
2. **Start labeling**: Press `K` to mark as "keep", `T` for "trash"
3. **Watch the magic**: After a few labels, AI starts suggesting labels automatically
4. **Trust your gut**: The model learns from your decisions and gets better over time

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `K` | Mark as **Keep** |
| `T` | Mark as **Trash** |
| `1-5` | Rate image (1=worst, 5=best) |
| `←` `→` | Navigate between images |
| `?` | Show all shortcuts |

**Pro tip**: The app is designed for keyboard-only use. Once you get the hang of it, you can label hundreds of photos in minutes!

---

## 🧠 How the AI Works

Photo Derush uses a **CatBoost classifier** trained on:

- **78 handcrafted features**: EXIF data, image quality metrics, histograms, and more
- **128 embedding features**: ResNet18 visual embeddings (PCA-reduced)
- **Object detection**: YOLOv8 for detecting people, objects, and scenes

The model is optimized for **asymmetric costs**: it's much better at avoiding false negatives (marking a good photo as trash) than false positives. This means you'll rarely lose a photo you want to keep.

### Model Performance

- **Accuracy**: 75.31% ± 2.51% (5-fold cross-validation)
- **Keep-Loss Rate**: 1.03% ± 0.42% ✅ (only 1% of good photos misclassified as trash)
- **PR-AUC**: 0.9290 (excellent precision-recall balance)
- **Training Time**: < 1 second for fast mode (interactive retraining)

> 📖 **Want the technical details?** See [docs/MODEL.md](docs/MODEL.md) for full model documentation, methodology, and evaluation metrics.

---

## 🎨 Features

### Core Functionality

- **📁 Smart Photo Browser**: Grid view with thumbnails, sorting, and filtering
- **🏷️ Flexible Labeling**: Keep/trash labels, 1-5 star ratings, and custom tags
- **🤖 Auto-Labeling**: AI suggests labels with confidence thresholds
- **🔄 Real-Time Retraining**: Model updates as you label (under 5 seconds per update)
- **👁️ Object Detection**: Visualize detected objects with bounding boxes
- **📊 Model Stats**: See accuracy, F1 score, and feature importance in real-time
- **🌙 Dark Theme**: Easy on the eyes for long labeling sessions

### Advanced Features

- **Duplicate Detection**: Perceptual hashing to group near-duplicate photos
- **Uncertainty Sorting**: See which photos the model is least confident about
- **Batch Operations**: Label multiple images at once
- **EXIF Viewer**: Camera settings, GPS data, and more
- **Fullscreen Viewer**: Press `F` to view images fullscreen with overlay controls

---

## 🛠️ Development

### Setup

```bash
# Install dependencies
poetry install

# Enter virtual environment
poetry shell

# Run tests
pytest tests/

# Run with debug logging
poetry run python app.py --log-file debug.log
```

### Project Structure

```
photo-derush/
├── src/              # Main application code
│   ├── view.py       # UI (PySide6/Qt)
│   ├── viewmodel.py  # Business logic
│   ├── training_core.py  # ML training pipeline
│   └── inference.py  # Model inference
├── scripts/          # Training and analysis scripts
├── docs/             # Documentation
│   ├── MODEL.md      # Model documentation
│   └── technical_report.md  # Full technical details
└── tests/            # Test suite
```

### Tech Stack

- **Desktop UI**: PySide6 (Qt)
- **ML Framework**: CatBoost + scikit-learn
- **Object Detection**: YOLOv8
- **Embeddings**: ResNet18 (via torchvision)
- **Database**: SQLite (for labels and metadata)

---

## 📚 Documentation

- **[Model Documentation](docs/MODEL.md)**: Best model configuration, metrics, and methodology
- **[Technical Report](docs/technical_report.md)**: Comprehensive technical details and evaluation
- **[Improvements Roadmap](docs/IMPROVEMENTS.md)**: Future enhancements and ideas
- **[Studies](docs/studies/)**: Learning rate studies, AVA dataset experiments
- **[Features](docs/features/)**: Embeddings, PCA, dataset documentation

---

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

- 🐛 **Report bugs**: Open an issue with details
- 💡 **Suggest features**: Share your ideas
- 🔧 **Submit PRs**: Fix bugs or add features
- 📖 **Improve docs**: Make the documentation better
- ⭐ **Star the repo**: Show your support!

### Development Guidelines

- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Keep commits focused and meaningful

---

## 🎯 Use Cases

- **📷 Photo Library Cleanup**: Sort through years of accumulated photos
- **🎬 Event Photography**: Quickly triage event photos before editing
- **📱 Phone Photo Management**: Clean up your phone's camera roll
- **🖼️ Stock Photo Curation**: Organize and filter stock photo collections
- **📸 Professional Workflow**: Pre-sort photos before post-processing

---

## ⚙️ Configuration

The app stores configuration in `~/.photo_app_config.json`. You can customize:

- Default image directory
- Auto-labeling thresholds
- Model paths
- UI preferences

---

## 🐛 Troubleshooting

**App won't start?**
- Make sure you have Python 3.12+ installed
- Check that all dependencies are installed: `poetry install`
- Try running with `--log-file` to see error messages

**Model predictions seem wrong?**
- Label more photos—the model improves with more training data
- Check model stats in the sidebar to see current accuracy
- The model retrains automatically after each label change

**Performance issues?**
- Large image directories (>10k images) may be slow on first load
- Consider using the fast mode for retraining (enabled by default)
- Check the logs for specific bottlenecks

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- Built with [PySide6](https://www.qt.io/qt-for-python) for the desktop UI
- Powered by [CatBoost](https://catboost.ai/) for machine learning
- Uses [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- Inspired by the need to actually organize our photo collections

---

## 💬 Questions?

- Open an issue for bugs or feature requests
- Check the [documentation](docs/) for detailed information
- Review the [technical report](docs/technical_report.md) for ML methodology

---

**Made with ❤️ for photographers, organizers, and anyone drowning in photos**

*Photo Derush - Because life's too short to manually sort 10,000 photos* 📸✨
