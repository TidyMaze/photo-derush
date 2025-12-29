from __future__ import annotations

"""Model statistics formatting (SRP)."""


def _get_feature_names(meta: dict | None = None) -> dict:
    """Build complete feature name mapping for FAST mode (78 features) + interactions + embeddings.
    
    Args:
        meta: Optional model metadata to determine interaction/embedding feature indices
    """
    names = {}
    # 0-11: Basic stats
    names.update(
        {
            0: "Width",
            1: "Height",
            2: "Aspect Ratio",
            3: "File Size (log)",
            4: "Mean Brightness",
            5: "Std Brightness",
            6: "Mean Red",
            7: "Mean Green",
            8: "Mean Blue",
            9: "Std Red",
            10: "Std Green",
            11: "Std Blue",
        }
    )
    # 12-35: RGB histograms (8 bins per channel)
    for ch, color in enumerate(["R", "G", "B"]):
        for bin in range(8):
            names[12 + ch * 8 + bin] = f"{color} Hist Bin {bin}"
    # 36-42: Image quality metrics
    names.update(
        {
            36: "Sharpness",
            37: "Saturation",
            38: "Entropy",
            39: "Std Brightness (dup)",
            40: "Highlight Clip",
            41: "Shadow Clip",
            42: "Noise Level",
        }
    )
    # 43-46: Temporal features
    names.update({43: "Hour of Day", 44: "Day of Week", 45: "Month", 46: "Is Weekend"})
    # 47-56: EXIF features
    names.update(
        {
            47: "ISO",
            48: "Aperture",
            49: "Shutter Speed",
            50: "Flash Fired",
            51: "Focal Length 35mm",
            52: "Digital Zoom",
            53: "Exposure Compensation",
            54: "White Balance",
            55: "Exposure Mode",
            56: "Metering Mode",
        }
    )
    # 57-62: Advanced features (stubbed)
    names.update(
        {
            57: "Edge Density",
            58: "Edge Strength",
            59: "Corner Count",
            60: "Face Count",
            61: "Histogram Balance",
            62: "Color Temperature",
        }
    )
    # 63-76: Additional advanced features
    names.update(
        {
            63: "Center Brightness Ratio",
            64: "Exposure Quality",
            65: "Color Diversity",
            66: "Rule of Thirds Score",
            67: "Symmetry Score",
            68: "Horizon Levelness",
            69: "Center Focus Quality",
            70: "Dynamic Range Utilization",
            71: "Subject Isolation",
            72: "Golden Hour",
            73: "Lighting Quality",
            74: "Color Harmony",
            75: "Sky Ground Ratio",
            76: "Motion Blur",
        }
    )
    # 77: Person Detection
    names.update({77: "Person Detection"})
    
    # Add interaction and embedding feature names if metadata provided
    if meta:
        n_base = meta.get("n_base_features", 78)
        interaction_pairs = meta.get("interaction_pairs", [])
        ratio_pairs = meta.get("ratio_pairs", [])
        pca_dim = meta.get("pca_dim")
        
        # Add interaction features (after base features)
        idx = n_base
        for i, j in interaction_pairs:
            base_i = names.get(i, f"Feature {i}")
            base_j = names.get(j, f"Feature {j}")
            names[idx] = f"{base_i} × {base_j}"
            idx += 1
        
        # Add ratio features (after interactions)
        for i, j in ratio_pairs:
            base_i = names.get(i, f"Feature {i}")
            base_j = names.get(j, f"Feature {j}")
            names[idx] = f"{base_i} / {base_j}"
            idx += 1
        
        # Add embedding features (after interactions and ratios)
        if pca_dim:
            for emb_idx in range(pca_dim):
                names[idx] = f"Embedding {emb_idx}"
                idx += 1
    
    return names


def format_model_stats(stats: dict) -> str:
    if not stats:
        return ""
    # Get metadata from stats if available (for interaction/embedding feature names)
    meta = stats.get("model_metadata", {})
    feature_names = _get_feature_names(meta=meta)
    lines = []
    lines.append("═══ MODEL STATISTICS ═══")
    lines.append("")
    n_samples = stats.get("n_samples", 0)
    n_keep = stats.get("n_keep", 0)
    n_trash = stats.get("n_trash", 0)
    lines.append(f"Dataset: {n_samples} samples")
    if n_samples:
        lines.append(f"  • Keep: {n_keep} ({n_keep / n_samples * 100:.1f}%)")
        lines.append(f"  • Trash: {n_trash} ({n_trash / n_samples * 100:.1f}%)")
    else:
        lines.append("  • Keep: 0")
        lines.append("  • Trash: 0")
    lines.append("")
    precision = stats.get("precision")
    if precision is not None:
        lines.append(f"Precision: {precision * 100:.2f}%")
    cv_mean = stats.get("cv_accuracy_mean")
    cv_std = stats.get("cv_accuracy_std")
    if cv_mean is not None:
        if cv_std is not None:
            lines.append(f"CV Accuracy: {cv_mean * 100:.2f}% ± {cv_std * 100:.2f}%")
        else:
            lines.append(f"CV Accuracy: {cv_mean * 100:.2f}%")
    fi = stats.get("feature_importances")
    if fi:
        lines.append("")
        lines.append("Top 10 Features:")
        # Normalize importances to percentages (sum to 100%)
        total_importance = sum(imp for _, imp in fi) if fi else 1.0
        for idx, (feat_idx, importance) in enumerate(fi[:10], 1):
            percentage = (importance / total_importance * 100) if total_importance > 0 else 0.0
            lines.append(f'  {idx}. {feature_names.get(feat_idx, f"Feature {feat_idx}")}: {percentage:.2f}%')
    # Thresholds
    keep_base = stats.get("keep_threshold_base")
    trash_base = stats.get("trash_threshold_base")
    keep_eff = stats.get("keep_threshold_eff")
    trash_eff = stats.get("trash_threshold_eff")
    weighted_mode = stats.get("weighted_mode")
    if keep_base is not None and trash_base is not None:
        lines.append("")
        if keep_eff is not None and trash_eff is not None:
            lines.append(
                f'Thresholds: keep>={keep_eff:.2f} (base {keep_base:.2f}) | trash<={trash_eff:.2f} (base {trash_base:.2f}) | mode={"weighted" if weighted_mode else "fixed"}'
            )
        else:
            lines.append(f"Thresholds: keep>={keep_base:.2f} | trash<={trash_base:.2f} | mode=fixed")
    lines.append("")
    mp = stats.get("model_path", "")
    if mp:
        import os

        lines.append(f"Model: {os.path.basename(mp)}")
    return "\n".join(lines)


__all__ = ["format_model_stats"]
