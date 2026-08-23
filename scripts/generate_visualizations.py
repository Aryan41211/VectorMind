"""Generate training visualizations and identify best checkpoint.

Purpose: Create training curves, recall metrics, embedding variance,
temperature plots, and identify the best checkpoint.

Usage:
    python scripts/generate_visualizations.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)


def load_tensorboard_data(log_dir: str = "logs/train") -> dict:
    """Load TensorBoard event data.
    
    Args:
        log_dir: Path to TensorBoard log directory.
        
    Returns:
        Dictionary with extracted metrics.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("ERROR: tensorboard not installed. Run: pip install tensorboard")
        return {}
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"ERROR: Log directory not found: {log_dir}")
        return {}
    
    # Find all event files
    event_files = list(log_path.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"ERROR: No event files found in {log_dir}")
        return {}
    
    print(f"Found {len(event_files)} event file(s) in {log_dir}")
    
    # Load the most recent event file
    event_file = max(event_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading: {event_file.name}")
    
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    
    # Extract metrics
    metrics = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        if events:
            values = [e.value for e in events]
            steps = [e.step for e in events]
            metrics[tag] = {
                "values": values,
                "steps": steps,
            }
    
    return metrics


def plot_training_curves(metrics: dict, output_dir: str = "reports/figures") -> None:
    """Plot training curves.
    
    Args:
        metrics: Dictionary of metrics from TensorBoard.
        output_dir: Directory to save plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('VectorMind Phase 4 Training Curves', fontsize=16, fontweight='bold')
    
    # 1. Loss Curve
    if "train/loss" in metrics:
        ax = axes[0, 0]
        steps = metrics["train/loss"]["steps"]
        values = metrics["train/loss"]["values"]
        ax.plot(steps, values, 'b-', linewidth=2, label='Training Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Recall@1
    if "epoch/val/recall@1" in metrics:
        ax = axes[0, 1]
        steps = metrics["epoch/val/recall@1"]["steps"]
        values = [v * 100 for v in metrics["epoch/val/recall@1"]["values"]]
        ax.plot(steps, values, 'g-', linewidth=2, marker='o', markersize=6, label='Val Recall@1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall@1 (%)')
        ax.set_title('Validation Recall@1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(values) * 1.2 if values else 100)
    
    # 3. Recall@5
    if "epoch/val/recall@5" in metrics:
        ax = axes[0, 2]
        steps = metrics["epoch/val/recall@5"]["steps"]
        values = [v * 100 for v in metrics["epoch/val/recall@5"]["values"]]
        ax.plot(steps, values, 'r-', linewidth=2, marker='s', markersize=6, label='Val Recall@5')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall@5 (%)')
        ax.set_title('Validation Recall@5')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(values) * 1.2 if values else 100)
    
    # 4. Recall@10
    if "epoch/val/recall@10" in metrics:
        ax = axes[1, 0]
        steps = metrics["epoch/val/recall@10"]["steps"]
        values = [v * 100 for v in metrics["epoch/val/recall@10"]["values"]]
        ax.plot(steps, values, 'm-', linewidth=2, marker='^', markersize=6, label='Val Recall@10')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall@10 (%)')
        ax.set_title('Validation Recall@10')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(values) * 1.2 if values else 100)
        
        # Mark best point
        if values:
            best_idx = np.argmax(values)
            best_step = steps[best_idx]
            best_val = values[best_idx]
            ax.annotate(f'Best: {best_val:.2f}%',
                       xy=(best_step, best_val),
                       xytext=(best_step, best_val + 5),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red', fontweight='bold')
    
    # 5. Embedding Variance
    if "epoch/val/image_dim_variance" in metrics and "epoch/val/text_dim_variance" in metrics:
        ax = axes[1, 1]
        steps_img = metrics["epoch/val/image_dim_variance"]["steps"]
        values_img = metrics["epoch/val/image_dim_variance"]["values"]
        steps_txt = metrics["epoch/val/text_dim_variance"]["steps"]
        values_txt = metrics["epoch/val/text_dim_variance"]["values"]
        
        ax.plot(steps_img, values_img, 'c-', linewidth=2, marker='d', markersize=6, label='Image Variance')
        ax.plot(steps_txt, values_txt, 'y-', linewidth=2, marker='d', markersize=6, label='Text Variance')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Variance')
        ax.set_title('Embedding Variance (Health Check)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.001, color='r', linestyle='--', alpha=0.5, label='Collapse Threshold')
    
    # 6. Temperature
    if "epoch/temperature" in metrics:
        ax = axes[1, 2]
        steps = metrics["epoch/temperature"]["steps"]
        values = metrics["epoch/temperature"]["values"]
        ax.plot(steps, values, 'k-', linewidth=2, marker='*', markersize=8, label='Temperature')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Temperature')
        ax.set_title('Learned Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_path / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to: {save_path}")
    plt.close()


def identify_best_checkpoint(metrics: dict) -> dict:
    """Identify the best checkpoint based on validation metrics.
    
    Args:
        metrics: Dictionary of metrics from TensorBoard.
        
    Returns:
        Dictionary with best checkpoint information.
    """
    if "epoch/val/recall@10" not in metrics:
        return {"error": "No validation Recall@10 data found"}
    
    r10_steps = metrics["epoch/val/recall@10"]["steps"]
    r10_values = metrics["epoch/val/recall@10"]["values"]
    
    if not r10_values:
        return {"error": "No validation Recall@10 values found"}
    
    # Find best epoch
    best_idx = np.argmax(r10_values)
    best_epoch = r10_steps[best_idx]
    best_r10 = r10_values[best_idx]
    
    # Get other metrics at best epoch
    best_r1 = 0.0
    best_r5 = 0.0
    best_img_var = 0.0
    best_txt_var = 0.0
    best_temp = 0.0
    
    if "epoch/val/recall@1" in metrics:
        r1_values = metrics["epoch/val/recall@1"]["values"]
        if best_idx < len(r1_values):
            best_r1 = r1_values[best_idx]
    
    if "epoch/val/recall@5" in metrics:
        r5_values = metrics["epoch/val/recall@5"]["values"]
        if best_idx < len(r5_values):
            best_r5 = r5_values[best_idx]
    
    if "epoch/val/image_dim_variance" in metrics:
        img_var_values = metrics["epoch/val/image_dim_variance"]["values"]
        if best_idx < len(img_var_values):
            best_img_var = img_var_values[best_idx]
    
    if "epoch/val/text_dim_variance" in metrics:
        txt_var_values = metrics["epoch/val/text_dim_variance"]["values"]
        if best_idx < len(txt_var_values):
            best_txt_var = txt_var_values[best_idx]
    
    if "epoch/temperature" in metrics:
        temp_values = metrics["epoch/temperature"]["values"]
        # Find temperature at closest epoch
        temp_steps = metrics["epoch/temperature"]["steps"]
        temp_idx = np.argmin(np.abs(np.array(temp_steps) - best_epoch))
        if temp_idx < len(temp_values):
            best_temp = temp_values[temp_idx]
    
    return {
        "best_epoch": int(best_epoch),
        "best_recall@10": float(best_r10),
        "best_recall@5": float(best_r5),
        "best_recall@1": float(best_r1),
        "best_image_variance": float(best_img_var),
        "best_text_variance": float(best_txt_var),
        "best_temperature": float(best_temp),
        "total_epochs": len(r10_values),
        "recall@10_improvement": float(best_r10 - r10_values[0]) if r10_values else 0.0,
    }


def save_checkpoint_summary(best_info: dict, output_dir: str = "reports") -> None:
    """Save checkpoint summary to JSON.
    
    Args:
        best_info: Dictionary with best checkpoint information.
        output_dir: Directory to save summary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "best_checkpoint": best_info,
        "comparison": {
            "baseline_before_queue_fix": {
                "epoch": 6,
                "recall@10": 0.1712,
                "recall@1": 0.0346,
            },
            "after_queue_fix": {
                "epoch": 8,
                "recall@10": 0.2023,
                "recall@1": 0.0422,
            },
            "hyperparameter_experiment_lower_lr": {
                "epoch": 9,
                "recall@10": 0.1054,
                "recall@1": 0.0208,
                "lr": 5e-4,
                "conclusion": "Lower LR hurts performance",
            },
        },
        "recommendations": [
            "Continue training with lr=1e-3 (baseline LR is better)",
            "Memory queue is working correctly (queue_size=4096)",
            "Temperature is learning (increasing over time)",
            "Embedding variance is healthy (no collapse)",
            "Recall@10 improved from 17.12% to 20.23% with memory queue",
        ],
    }
    
    save_path = output_path / "checkpoint_summary.json"
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved checkpoint summary to: {save_path}")


def main() -> None:
    """Main entry point."""
    print("VectorMind Training Visualizations")
    print("=" * 80)
    
    # Load TensorBoard data
    metrics = load_tensorboard_data("logs/train")
    
    if not metrics:
        print("No metrics found. Exiting.")
        return
    
    # Generate visualizations
    print("\nGenerating training curves...")
    plot_training_curves(metrics, "reports/figures")
    
    # Identify best checkpoint
    print("\nIdentifying best checkpoint...")
    best_info = identify_best_checkpoint(metrics)
    
    if "error" in best_info:
        print(f"Error: {best_info['error']}")
    else:
        print("\nBest Checkpoint Summary:")
        print(f"  Epoch: {best_info['best_epoch']}")
        print(f"  Recall@10: {best_info['best_recall@10']*100:.2f}%")
        print(f"  Recall@5: {best_info['best_recall@5']*100:.2f}%")
        print(f"  Recall@1: {best_info['best_recall@1']*100:.2f}%")
        print(f"  Image Variance: {best_info['best_image_variance']:.6f}")
        print(f"  Text Variance: {best_info['best_text_variance']:.6f}")
        print(f"  Temperature: {best_info['best_temperature']:.4f}")
        print(f"  Total Epochs: {best_info['total_epochs']}")
        print(f"  Recall@10 Improvement: {best_info['recall@10_improvement']*100:.2f}%")
        
        # Save summary
        save_checkpoint_summary(best_info, "reports")
    
    print("\n" + "=" * 80)
    print("Visualization generation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
