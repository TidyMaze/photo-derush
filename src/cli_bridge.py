#!/usr/bin/env python3
"""
CLI Bridge for Darktable Lua Plugin.
Provides JSON-RPC style interface to expose photo-derush ML, grouping, and feature extraction.
"""

import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure log file in Darktable AppData folder
DARKTABLE_DIR = os.path.join(os.getenv("LOCALAPPDATA", os.path.expanduser("~")), "darktable")
os.makedirs(DARKTABLE_DIR, exist_ok=True)
PYTHON_LOG_PATH = os.path.join(DARKTABLE_DIR, "derush_python.log")

file_handler = logging.FileHandler(PYTHON_LOG_PATH, mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers = [file_handler, stderr_handler]

from src.features import extract_features
from src.grouping_service import compute_grouping_for_photos
from src.image_cache import normalize_path
from src.model import ImageModel as PhotoModel
from src.repository import RatingsTagsRepository


def cmd_scan(args):
    """Scan folder and extract basic photo stats."""
    directory = args.directory
    model = PhotoModel(directory=directory, max_images=None)
    image_names = model.get_image_files()
    output = {
        "status": "success",
        "directory": directory,
        "count": len(image_names),
        "images": image_names[:100]
    }
    print(json.dumps(output))


def cmd_group(args):
    """Run pHash burst grouping on directory."""
    directory = args.directory
    model = PhotoModel(directory=directory, max_images=None)
    image_names = model.get_image_files()

    # Preload EXIF for grouping calculation
    exif_data = {fn: model.load_exif(model.get_image_path(fn)) for fn in image_names if model.get_image_path(fn)}
    group_info = compute_grouping_for_photos(
        filenames=image_names,
        image_dir=directory,
        exif_data=exif_data
    )

    output = {
        "status": "success",
        "total_images": len(image_names),
        "groups": group_info
    }
    print(json.dumps(output))


def cmd_predict(args_or_dir, labels_file=None, files_json=None, burst_limit=False):
    """Predict keep probabilities using ML classifier and compute burst groups."""
    if isinstance(args_or_dir, str):
        directory = args_or_dir
        file_list = json.loads(files_json) if files_json else None
    else:
        directory = args_or_dir.directory
        file_list = None
        if getattr(args_or_dir, "files_file", None) and os.path.exists(args_or_dir.files_file):
            try:
                with open(args_or_dir.files_file, "r", encoding="utf-8") as f:
                    file_list = json.load(f)
            except Exception as e:
                sys.stderr.write(f"Failed reading files file: {e}\n")
        burst_limit = getattr(args_or_dir, "burst_limit", False) or burst_limit

    directory = normalize_path(directory) if directory else directory
    model = PhotoModel(directory=directory, max_images=None)
    if file_list:
        image_names = [os.path.basename(f) for f in file_list]
        image_paths = [model.get_image_path(f) for f in file_list]
    else:
        image_names = model.get_image_files()
        image_paths = [model.get_image_path(fn) for fn in image_names if model.get_image_path(fn)]

    RAW_EXTS = {".arw", ".cr2", ".nef", ".dng", ".orf", ".rw2", ".pef", ".srw"}
    stem_map = {}
    for fn, path in zip(image_names, image_paths):
        if not path:
            continue
        stem = os.path.splitext(fn)[0]
        if stem not in stem_map:
            stem_map[stem] = (fn, path)
        else:
            prev_fn, prev_path = stem_map[stem]
            prev_ext = os.path.splitext(prev_fn)[1].lower()
            curr_ext = os.path.splitext(fn)[1].lower()
            if prev_ext in RAW_EXTS and curr_ext not in RAW_EXTS:
                stem_map[stem] = (fn, path)

    stems = list(stem_map.keys())
    unique_paths = [stem_map[s][1] for s in stems]

    probs = {}
    try:
        from src.inference import predict_keep_probability
        probabilities = predict_keep_probability(unique_paths)
        for stem, prob in zip(stems, probabilities):
            if prob == prob:  # check for nan
                p_val = round(float(prob), 4)
                probs[stem] = p_val
                probs[stem.lower()] = p_val
    except Exception as e:
        sys.stderr.write(f"Prediction error: {e}\n")

    # Propagate predictions to all files (RAW + JPG) and paths matching the stem
    for fn, path in zip(image_names, image_paths):
        stem = os.path.splitext(fn)[0]
        if stem in probs:
            p_val = probs[stem]
            probs[fn] = p_val
            probs[fn.lower()] = p_val
            if path:
                probs[path] = p_val
                probs[path.lower()] = p_val
                probs[os.path.basename(path)] = p_val
                probs[os.path.basename(path).lower()] = p_val

    decision_threshold = 0.50
    try:
        from src.inference import load_model
        bundle = load_model()
        if bundle and bundle.meta:
            raw_thresh = bundle.meta.get("decision_threshold")
            if raw_thresh is None:
                raw_thresh = bundle.meta.get("keep_ratio", 0.50)
            if raw_thresh and raw_thresh > 0:
                decision_threshold = float(raw_thresh)
    except Exception:
        pass

    # Compute grouping for images
    group_info = {}
    try:
        exif_data = {fn: model.load_exif(p) for fn, p in zip(image_names, image_paths) if p and os.path.exists(p)}
        group_info = compute_grouping_for_photos(
            filenames=image_names,
            image_dir=directory,
            exif_data=exif_data,
            keep_probabilities=probs,
            burst_gap_sec=2.0
        )
    except Exception as e:
        sys.stderr.write(f"Grouping computation error: {e}\n")

    # Propagate group_info to stems, lowercase, and paired RAW/JPG files
    from src.image_cache import find_paired_jpg
    propagated_groups = {}
    for fn, g_data in group_info.items():
        stem = os.path.splitext(fn)[0]
        propagated_groups[fn] = g_data
        propagated_groups[fn.lower()] = g_data
        propagated_groups[stem] = g_data
        propagated_groups[stem.lower()] = g_data

        path = model.get_image_path(fn)
        if path:
            paired = find_paired_jpg(path)
            if paired:
                paired_fn = os.path.basename(paired)
                propagated_groups[paired_fn] = g_data
                propagated_groups[paired_fn.lower()] = g_data
                paired_stem = os.path.splitext(paired_fn)[0]
                propagated_groups[paired_stem] = g_data
                propagated_groups[paired_stem.lower()] = g_data

    output = {
        "status": "success",
        "total_images": len(image_names),
        "threshold": round(float(decision_threshold), 4),
        "predictions": probs,
        "groups": propagated_groups
    }
    result_str = json.dumps(output)
    if isinstance(args_or_dir, str):
        return result_str
    print(result_str)


def cmd_train(args):
    """Save manual labels from Darktable and retrain the model."""
    directory = args.directory
    labels_file = os.path.join(directory, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=labels_file)

    labels_json_data = args.labels_json
    if args.labels_file and os.path.exists(args.labels_file):
        try:
            with open(args.labels_file, "r", encoding="utf-8") as f:
                labels_json_data = f.read()
        except Exception as e:
            sys.stderr.write(f"Failed reading labels file: {e}\n")

    if labels_json_data:
        try:
            user_labels = json.loads(labels_json_data)
            for fn, state in user_labels.items():
                repo.set_state(fn, state)
        except Exception as e:
            sys.stderr.write(f"Failed to parse labels JSON: {e}\n")

    file_list = None
    if getattr(args, "files_file", None) and os.path.exists(args.files_file):
        try:
            with open(args.files_file, "r", encoding="utf-8") as f:
                file_list = json.load(f)
        except Exception as e:
            sys.stderr.write(f"Failed reading files file: {e}\n")

    # Retrain model
    retrain_result = {}
    try:
        from src.training_core import train_keep_trash_model
        model_path = os.path.expanduser("~/.photo-derush-keep-trash-model.joblib")
        res = train_keep_trash_model(directory, repo=repo, model_path=model_path, fast_mode=True, displayed_filenames=file_list)
        if res:
            retrain_result = {
                "status": "success",
                "model_path": res.model_path,
                "n_samples": res.n_samples,
                "n_keep": res.n_keep,
                "n_trash": res.n_trash,
                "cv_accuracy_mean": res.cv_accuracy_mean,
                "accuracy": res.accuracy,
                "precision": res.precision,
                "roc_auc": res.roc_auc,
                "f1": res.f1,
            }
        else:
            retrain_result = {"status": "warning", "message": "Insufficient data to train"}
    except Exception as e:
        sys.stderr.write(f"Retrain error: {e}\n")
        retrain_result = {"status": "error", "message": str(e)}

    output = {
        "status": "success",
        "result": retrain_result
    }
    print(json.dumps(output))


def main():
    parser = argparse.ArgumentParser(description="Derush Darktable CLI Bridge")
    subparsers = parser.add_subparsers(dest="command")

    # Scan command
    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--directory", required=False, help="Path to photo directory")
    scan_parser.add_argument("--directory-file", required=False, help="Path to file containing directory path")

    # Group command
    group_parser = subparsers.add_parser("group")
    group_parser.add_argument("--directory", required=False, help="Path to photo directory")
    group_parser.add_argument("--directory-file", required=False, help="Path to file containing directory path")

    # Predict command
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--directory", required=False, help="Path to photo directory")
    predict_parser.add_argument("--directory-file", required=False, help="Path to file containing directory path")
    predict_parser.add_argument("--files-file", required=False, help="Path to JSON file containing list of image paths")

    # Train command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--directory", required=False, help="Path to photo directory")
    train_parser.add_argument("--directory-file", required=False, help="Path to file containing directory path")
    train_parser.add_argument("--labels-json", required=False, help="JSON map of filename -> keep/trash state")
    train_parser.add_argument("--labels-file", required=False, help="Path to JSON file containing labels")
    train_parser.add_argument("--files-file", required=False, help="Path to JSON file containing list of image paths")

    args = parser.parse_args()
    logging.info(f"=== CLI BRIDGE COMMAND: {args.command} | PID: {os.getpid()} ===")

    # Resolve directory from file if --directory-file is provided
    directory_file = getattr(args, "directory_file", None)
    if directory_file and os.path.isfile(directory_file):
        with open(directory_file, "r", encoding="utf-8") as f:
            args.directory = f.read().strip()

    if not getattr(args, "directory", None) and args.command in ("scan", "group", "predict", "train"):
        parser.error("Either --directory or --directory-file is required")

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "group":
        cmd_group(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
