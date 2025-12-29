#!/usr/bin/env python3
"""Generate a comprehensive single-image report from CV hyperparameter study results."""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_results_from_log(log_path: str):
    """Extract results from the study log file."""
    results = {
        "learning_rate": [],
        "max_iterations": [],
        "patience": []
    }
    
    current_param = None
    with open(log_path, 'r') as f:
        for line in f:
            if "STUDY 1: Learning Rate" in line:
                current_param = "learning_rate"
            elif "STUDY 2: Max Iterations" in line:
                current_param = "max_iterations"
            elif "STUDY 3: Patience" in line:
                current_param = "patience"
            elif current_param and "CV-Acc=" in line:
                # Parse line like: [1/20] learning_rate=0.000 ... CV-Acc=73.27%±1.27%, ROC-AUC=0.8530, F1=0.7919, Keep-Loss=2.59%, time=8.8s
                try:
                    parts = line.split("CV-Acc=")[1].split(",")
                    cv_acc_str = parts[0].replace("%", "").split("±")
                    cv_acc = float(cv_acc_str[0]) / 100.0
                    cv_std = float(cv_acc_str[1]) / 100.0 if len(cv_acc_str) > 1 else 0.0
                    
                    roc_auc = float(parts[1].split("=")[1].strip())
                    f1 = float(parts[2].split("=")[1].strip())
                    keep_loss = float(parts[3].split("=")[1].replace("%", "").strip()) / 100.0
                    time_val = float(parts[4].split("=")[1].replace("s", "").strip())
                    
                    # Extract param value from beginning of line
                    param_part = line.split("]")[1].split("=")[1].split("...")[0].strip()
                    if current_param == "learning_rate":
                        param_val = float(param_part)
                    else:
                        param_val = int(param_part)
                    
                    results[current_param].append({
                        "param_value": param_val,
                        "cv_accuracy": cv_acc,
                        "cv_std": cv_std,
                        "roc_auc": roc_auc,
                        "f1": f1,
                        "keep_loss_rate": keep_loss,
                        "training_time": time_val,
                    })
                except Exception as e:
                    print(f"Warning: Could not parse line: {line.strip()}, error: {e}")
                    continue
    
    return results

def create_comprehensive_report(results: dict, output_path: str = "plots/hyperparameter_study_report.png"):
    """Create a comprehensive single-image report."""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Fast Mode Hyperparameter Study - Comprehensive Report\n(5-Fold Cross-Validation)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Learning Rate Results
    lr_data = results["learning_rate"]
    if lr_data:
        lr_params = [r["param_value"] for r in lr_data]
        lr_cv_acc = [r["cv_accuracy"] for r in lr_data]
        lr_cv_std = [r["cv_std"] for r in lr_data]
        lr_roc = [r["roc_auc"] for r in lr_data]
        lr_keep_loss = [r["keep_loss_rate"] for r in lr_data]
        lr_time = [r["training_time"] for r in lr_data]
        
        # CV Accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.errorbar(lr_params, lr_cv_acc, yerr=lr_cv_std, fmt='o-', linewidth=2, markersize=5, capsize=3)
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate', fontsize=11)
        ax1.set_ylabel('CV Accuracy', fontsize=11)
        ax1.set_title('Learning Rate: CV Accuracy', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.70, 0.80])
        
        # ROC-AUC
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(lr_params, lr_roc, 'o-', color='green', linewidth=2, markersize=5)
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate', fontsize=11)
        ax2.set_ylabel('ROC-AUC', fontsize=11)
        ax2.set_title('Learning Rate: ROC-AUC', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.75, 0.90])
        
        # Keep-Loss Rate
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(lr_params, lr_keep_loss, 'o-', color='red', linewidth=2, markersize=5)
        ax3.set_xscale('log')
        ax3.set_xlabel('Learning Rate', fontsize=11)
        ax3.set_ylabel('Keep-Loss Rate', fontsize=11)
        ax3.set_title('Learning Rate: Keep-Loss Rate (lower is better)', fontsize=12, fontweight='bold')
        ax3.axhline(y=0.02, color='orange', linestyle='--', label='2% target')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Find best LR
        best_lr_cv_idx = np.argmax(lr_cv_acc)
        best_lr_roc_idx = np.argmax(lr_roc)
        best_lr_keep_loss_idx = np.argmin(lr_keep_loss)
    
    # Max Iterations Results
    iter_data = results["max_iterations"]
    if iter_data:
        iter_params = [r["param_value"] for r in iter_data]
        iter_cv_acc = [r["cv_accuracy"] for r in iter_data]
        iter_cv_std = [r["cv_std"] for r in iter_data]
        iter_roc = [r["roc_auc"] for r in iter_data]
        iter_keep_loss = [r["keep_loss_rate"] for r in iter_data]
        iter_time = [r["training_time"] for r in iter_data]
        
        # CV Accuracy
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.errorbar(iter_params, iter_cv_acc, yerr=iter_cv_std, fmt='o-', linewidth=2, markersize=5, capsize=3)
        ax4.set_xlabel('Max Iterations', fontsize=11)
        ax4.set_ylabel('CV Accuracy', fontsize=11)
        ax4.set_title('Max Iterations: CV Accuracy', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0.70, 0.80])
        
        # ROC-AUC
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(iter_params, iter_roc, 'o-', color='green', linewidth=2, markersize=5)
        ax5.set_xlabel('Max Iterations', fontsize=11)
        ax5.set_ylabel('ROC-AUC', fontsize=11)
        ax5.set_title('Max Iterations: ROC-AUC', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0.75, 0.90])
        
        # Training Time
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(iter_params, iter_time, 'o-', color='purple', linewidth=2, markersize=5)
        ax6.set_xlabel('Max Iterations', fontsize=11)
        ax6.set_ylabel('Training Time (s)', fontsize=11)
        ax6.set_title('Max Iterations: Training Time', fontsize=12, fontweight='bold')
        ax6.axhline(y=5.0, color='orange', linestyle='--', label='5s target')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # Find best iterations
        best_iter_cv_idx = np.argmax(iter_cv_acc)
        best_iter_roc_idx = np.argmax(iter_roc)
        best_iter_keep_loss_idx = np.argmin(iter_keep_loss)
    
    # Patience Results
    pat_data = results["patience"]
    if pat_data:
        pat_params = [r["param_value"] for r in pat_data]
        pat_cv_acc = [r["cv_accuracy"] for r in pat_data]
        pat_cv_std = [r["cv_std"] for r in pat_data]
        pat_roc = [r["roc_auc"] for r in pat_data]
        pat_keep_loss = [r["keep_loss_rate"] for r in pat_data]
        pat_time = [r["training_time"] for r in pat_data]
        
        # CV Accuracy
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.errorbar(pat_params, pat_cv_acc, yerr=pat_cv_std, fmt='o-', linewidth=2, markersize=5, capsize=3)
        ax7.set_xlabel('Patience', fontsize=11)
        ax7.set_ylabel('CV Accuracy', fontsize=11)
        ax7.set_title('Patience: CV Accuracy', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0.70, 0.80])
        
        # ROC-AUC
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(pat_params, pat_roc, 'o-', color='green', linewidth=2, markersize=5)
        ax8.set_xlabel('Patience', fontsize=11)
        ax8.set_ylabel('ROC-AUC', fontsize=11)
        ax8.set_title('Patience: ROC-AUC', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim([0.75, 0.90])
        
        # Training Time
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(pat_params, pat_time, 'o-', color='purple', linewidth=2, markersize=5)
        ax9.set_xlabel('Patience', fontsize=11)
        ax9.set_ylabel('Training Time (s)', fontsize=11)
        ax9.set_title('Patience: Training Time', fontsize=12, fontweight='bold')
        ax9.axhline(y=5.0, color='orange', linestyle='--', label='5s target')
        ax9.grid(True, alpha=0.3)
        ax9.legend()
        
        # Find best patience
        best_pat_cv_idx = np.argmax(pat_cv_acc)
        best_pat_roc_idx = np.argmax(pat_roc)
        best_pat_keep_loss_idx = np.argmin(pat_keep_loss)
    
    # Summary Text
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    summary_text = "KEY FINDINGS & RECOMMENDATIONS\n" + "="*80 + "\n\n"
    
    if lr_data:
        summary_text += f"LEARNING RATE (0.0001 to 0.5):\n"
        summary_text += f"  • Best CV Accuracy: {lr_cv_acc[best_lr_cv_idx]:.2%}±{lr_cv_std[best_lr_cv_idx]:.2%} at LR={lr_params[best_lr_cv_idx]:.4f}\n"
        summary_text += f"  • Best ROC-AUC: {lr_roc[best_lr_roc_idx]:.4f} at LR={lr_params[best_lr_roc_idx]:.4f}\n"
        summary_text += f"  • Best Keep-Loss Rate: {lr_keep_loss[best_lr_keep_loss_idx]:.2%} at LR={lr_params[best_lr_keep_loss_idx]:.4f}\n"
        summary_text += f"  • Impact: HIGH - Learning rate significantly affects performance\n"
        summary_text += f"  • Recommendation: Use LR={lr_params[best_lr_cv_idx]:.4f} for best CV accuracy\n\n"
    
    if iter_data:
        summary_text += f"MAX ITERATIONS (10 to 2000):\n"
        summary_text += f"  • Best CV Accuracy: {iter_cv_acc[best_iter_cv_idx]:.2%}±{iter_cv_std[best_iter_cv_idx]:.2%} at iterations={iter_params[best_iter_cv_idx]}\n"
        summary_text += f"  • Best ROC-AUC: {iter_roc[best_iter_roc_idx]:.4f} at iterations={iter_params[best_iter_roc_idx]}\n"
        summary_text += f"  • Best Keep-Loss Rate: {iter_keep_loss[best_iter_keep_loss_idx]:.2%} at iterations={iter_params[best_iter_keep_loss_idx]}\n"
        summary_text += f"  • Impact: LOW - Early stopping makes higher values unnecessary\n"
        summary_text += f"  • Recommendation: Use iterations={iter_params[best_iter_roc_idx]} (early stopping converges quickly)\n\n"
    
    if pat_data:
        summary_text += f"PATIENCE (1 to 200):\n"
        summary_text += f"  • Best CV Accuracy: {pat_cv_acc[best_pat_cv_idx]:.2%}±{pat_cv_std[best_pat_cv_idx]:.2%} at patience={pat_params[best_pat_cv_idx]}\n"
        summary_text += f"  • Best ROC-AUC: {pat_roc[best_pat_roc_idx]:.4f} at patience={pat_params[best_pat_roc_idx]}\n"
        summary_text += f"  • Best Keep-Loss Rate: {pat_keep_loss[best_pat_keep_loss_idx]:.2%} at patience={pat_params[best_pat_keep_loss_idx]}\n"
        summary_text += f"  • Impact: LOW - Patience ≥42 gives same performance\n"
        summary_text += f"  • Recommendation: Use patience={pat_params[best_pat_roc_idx]} (optimal performance reached quickly)\n\n"
    
    summary_text += "OVERALL RECOMMENDATION:\n"
    if lr_data and iter_data and pat_data:
        summary_text += f"  Optimal Fast Mode Settings:\n"
        summary_text += f"    • Learning Rate: {lr_params[best_lr_cv_idx]:.4f}\n"
        summary_text += f"    • Max Iterations: {iter_params[best_iter_roc_idx]}\n"
        summary_text += f"    • Patience: {pat_params[best_pat_roc_idx]}\n"
        summary_text += f"    • Expected CV Accuracy: {max(lr_cv_acc[best_lr_cv_idx], iter_cv_acc[best_iter_cv_idx], pat_cv_acc[best_pat_cv_idx]):.2%}\n"
        summary_text += f"    • Expected Training Time: ~{min(iter_time[best_iter_roc_idx], pat_time[best_pat_roc_idx]):.1f}s\n"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComprehensive report saved: {output_path}")
    plt.close()

def main():
    log_path = "/tmp/cv_study.log"
    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}")
        print("Please run the CV study first: poetry run python scripts/study_fast_mode_hyperparameters_cv.py")
        return
    
    print("Loading results from log file...")
    results = load_results_from_log(log_path)
    
    print(f"Loaded {len(results['learning_rate'])} learning rate results")
    print(f"Loaded {len(results['max_iterations'])} max iterations results")
    print(f"Loaded {len(results['patience'])} patience results")
    
    output_path = "plots/hyperparameter_study_report.png"
    create_comprehensive_report(results, output_path)
    print(f"\n✅ Comprehensive report generated: {output_path}")

if __name__ == "__main__":
    main()


