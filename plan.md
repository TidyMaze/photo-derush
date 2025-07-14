# 🗂️ Project TL;DR – “Derusher 9000”

> *Because scrolling through 2 000 near‑identical RAWs is beneath us.*

## 📸 What It Does

1. **Ingest & Backup** – copies every file (RAW/JPEG) to a dated backup folder *before* it dares touch a byte.
2. **Duplicate Slayer** – perceptual **dHash + FAISS** clusters near‑identical shots; keeps the sharpest / prettiest in each cluster.
3. **Blur & Dull Detector** –
   * **Blur** = Variance‑of‑Laplacian (OpenCV).
   * **Dullness** = NIMA aesthetic score (MobileNet‑V2 checkpoint).
4. **Auto‑Decision Rules** – default thresholds (blur < 180, aesthetic < 4, Hamming ≤ 5) or learned ones if a model exists.
5. **Soft Deletes** – files are merely **moved** into `workspace/trash/{duplicates,blurry,dull}`. Panic undo = drag ’em back.
6. **Feedback Loop** – every user override is stored in **SQLite**; a nightly (or on‑demand) retrain:
   * `StandardScaler ➜ GradientBoostingClassifier` (scikit‑learn)
   * New blur & aesthetic cut‑offs = 95th percentile of “trash” images.
7. **Active Learning** – uncertainty sampling so you label the *interesting* 1 % instead of 10 000 random frames.
8. **Typer CLI** – three arrogant commands:

   | Command     | What It Does                  | Example                                                             |
   | ----------- | ----------------------------- | ------------------------------------------------------------------- |
   | `ingest`    | backup + analyze + move trash | `derusher.py ingest ~/Shoot --workspace ~/Derush --backup ~/Backup` |
   | `train`     | retrain from overrides        | `derusher.py train --workspace ~/Derush`                            |
   | `uncertain` | list K most dubious images    | `derusher.py uncertain ~/Shoot --workspace ~/Derush -k 30`          |

## 🧰 Tech Stack (Why It Rules)

| Layer           | Lib / Tool                  | Why I Picked It                           |
| --------------- | --------------------------- | ----------------------------------------- |
| CLI             | **Typer**                   | Click‑style ergonomics, zero boilerplate. |
| Image IO        | **Pillow**, **rawpy**       | RAW → RGB in one line.                    |
| Perceptual Hash | **imagehash**               | Pure‑Python, fast enough.                 |
| ANN Search      | **faiss‑cpu**               | Millions of hashes? No sweat.             |
| Blur Metric     | **OpenCV**                  | The Laplacian trick everyone trusts.      |
| Aesthetic Score | **torch** + **timm**        | 30 ms per shot on CPU; faster on GPU.     |
| Model & AL      | **scikit‑learn**, **numpy** | Plain, dependable, no GPU drama.          |
| Storage         | **SQLite**                  | Zero‑config, ships with Python.           |
| Config          | JSON file                   | KISS; users can edit with Notepad.        |

## 🏗️ Minimalist Architecture

```
Typer CLI
   │
   ▼
Engine  ──► Analyzer (hash, blur, NIMA)
   │          │
   │          ▼
   │      FAISS index
   ▼
SQLite (feedback + models)
```

## 📈 Strengths

* **Fail‑safe** – hard backup + soft delete.
* **Self‑improving** – each override tightens thresholds & model weights.
* **Zero hard dependencies** – if Torch or OpenCV is missing, it degrades gracefully.

## 🔥 Limitations (a.k.a. “Stuff Future‑You Will Fix”)

1. Aesthetic model isn’t retrained per‑user yet (fine‑tune hook exists, not wired).
2. No GUI; CLI is fine for grown‑ups but your art‑director cousin will whine.
3. Face/subject weighting: right now a blurry picture of a rare smile still lands in trash.

## TODO

### Features to Implement Next

1. **Duplicate Slayer**
   - Integrate dHash and FAISS for clustering near-identical images.
   - Select sharpest/prettiest image in each cluster.
2. **Blur & Dull Detector**
   - Add OpenCV-based blur detection (Variance-of-Laplacian).
   - Integrate NIMA aesthetic scoring (MobileNet-V2 checkpoint).
3. **Auto-Decision Rules**
   - Apply default thresholds and support learned rules if model exists.
4. **Soft Deletes**
   - Move files to workspace/trash/{duplicates,blurry,dull} instead of deleting.
   - Implement undo (restore from trash).
5. **Feedback Loop**
   - Store user overrides in SQLite.
   - Nightly/on-demand retrain: StandardScaler ➜ GradientBoostingClassifier.
   - Update thresholds to 95th percentile of trash images.
6. **Active Learning**
   - Implement uncertainty sampling for efficient labeling.
7. **Typer CLI**
   - Add commands: ingest, train, uncertain.
   - Example usage and help output.
8. **Graceful Degradation**
   - If Torch or OpenCV missing, degrade features gracefully.
9. **Testing & Validation**
    - Add unit and integration tests for all major features.

### Planning
- Prioritize duplicate detection, then blur/dull analysis.
- CLI commands should be available early for workflow integration.
- Feedback loop and active learning can be added after core pipeline is stable.
- GUI and plugin API are future enhancements.
