from src.model_stats import format_model_stats


def test_format_model_stats_with_thresholds():
    stats = {
        'n_samples': 10,
        'n_keep': 6,
        'n_trash': 4,
        'precision': 0.8,
        'cv_accuracy_mean': 0.75,
        'cv_accuracy_std': 0.05,
        'model_path': '/tmp/model.joblib',
        'keep_threshold_base': 0.60,
        'trash_threshold_base': 0.40,
        'keep_threshold_eff': 0.65,
        'trash_threshold_eff': 0.40,
        'weighted_mode': True,
    }
    out = format_model_stats(stats)
    assert 'Thresholds:' in out
    assert 'keep>=0.65' in out
    assert '(base 0.60)' in out
    assert 'mode=weighted' in out

def test_format_model_stats_fixed_mode():
    stats = {
        'n_samples': 5,
        'n_keep': 3,
        'n_trash': 2,
        'precision': 0.9,
        'model_path': '/tmp/model.joblib',
        'keep_threshold_base': 0.70,
        'trash_threshold_base': 0.30,
        'weighted_mode': False,
    }
    out = format_model_stats(stats)
    assert 'thresholds:' .lower() in out.lower()
    assert 'mode=fixed' in out
    assert 'keep>=0.70' in out or 'keep>=0.70' in out.lower()

