# Study Script Outputs

All study/analysis scripts and their outputs are organized in the `studies/` directory:
- `studies/source/` - Analysis scripts (42 scripts)
- `studies/outputs/` - Generated images, HTML reports, and charts (21 files)

These scripts are standalone analysis/visualization tools and are not imported by the main application.

## Images and HTML files produced by study/analysis scripts

### Hash Correlation Analysis
- `studies/outputs/hash_correlation_matrix.png` - Full correlation matrix (591x591)
- `studies/outputs/hash_correlation_matrix_sampled.png` - Sampled matrix (296 images)
- `studies/outputs/hash_correlation_matrix_clustered.png` - Hierarchically clustered heatmap
- `studies/outputs/hash_correlation_matrix_network.png` - Similarity network graph
- `studies/outputs/hash_correlation_matrix_histogram.png` - Distance distribution histogram
- **Script**: `studies/source/plot_hash_correlation_matrix.py`

### Group Visualization
- `studies/outputs/top_10_groups.png` - Top 10 groups with all images
- **Script**: `studies/source/visualize_groups.py`

### Burst/Session Visualization
- `studies/outputs/bursts.png` - Images grouped by burst
- `studies/outputs/sessions.png` - Images grouped by session
- `studies/outputs/dates.png` - Images grouped by date
- `studies/outputs/bursts_sessions_hierarchical.png` - Hierarchical visualization (sessions → bursts → images)
- **Scripts**: 
  - `studies/source/visualize_bursts_sessions.py`
  - `studies/source/visualize_bursts_sessions_hierarchical.py`

### Hyperparameter Studies
- `studies/outputs/fast_mode_learning_rate_study.png` - Learning rate impact study
- `studies/outputs/fast_mode_learning_rate_study_cv.png` - Learning rate CV study
- `studies/outputs/fast_mode_max_iterations_study.png` - Max iterations study
- `studies/outputs/fast_mode_max_iterations_study_cv.png` - Max iterations CV study
- `studies/outputs/fast_mode_patience_study.png` - Patience study
- `studies/outputs/fast_mode_patience_study_cv.png` - Patience CV study
- `studies/outputs/hyperparameter_study_report.png` - Comprehensive hyperparameter report
- `studies/outputs/iterative_optimization_history.png` - Iterative optimization history
- **Scripts**:
  - `studies/source/study_fast_mode_hyperparameters.py`
  - `studies/source/study_fast_mode_hyperparameters_cv.py`
  - `studies/source/generate_hyperparameter_report.py`
  - `studies/source/iterative_hyperparameter_optimization.py`

### Learning Rate Impact
- `studies/outputs/learning_rate_impact.png` - Learning rate impact visualization
- `studies/outputs/learning_rate_impact_cv.png` - Learning rate impact with CV
- **Scripts**:
  - `studies/source/plot_learning_rate_impact.py`
  - `studies/source/study_learning_rate_impact.py`

### Technical Reports
- `studies/outputs/technical_report.html` - Technical report (HTML)
- `docs/technical_report.md` - Technical report (Markdown)
- **Script**: `studies/source/generate_technical_report.py`

### Model Evaluation Reports
- `studies/outputs/model_evaluation_report.html` - AVA model evaluation report (if generated)
- **Script**: `studies/source/evaluate_production_model_ava.py`

### Other Analysis Scripts (may produce outputs)
- `studies/source/analyze_group.py` - Group analysis (text output)
- `studies/source/compare_two_groups.py` - Group comparison (text output)
- `studies/source/check_group_separation.py` - Group separation analysis (text output)
- `studies/source/verify_grouping.py` - Grouping verification (text output)
- `studies/source/check_exif_dates.py` - EXIF date checking (text output)

### Screenshots
- `screenshots/main-interface.png` - Main application interface screenshot

### Test Images
- `test_images/*.jpg` - Test images for development

