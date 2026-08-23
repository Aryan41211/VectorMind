"""Analyze TensorBoard logs to extract training metrics.

Purpose: Parse TensorBoard event files and extract loss, recall, and other
metrics to understand the training run status.

Usage:
    python scripts/analyze_tensorboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def analyze_tensorboard_logs(log_dir: str = "logs/train") -> dict:
    """Parse TensorBoard event files and extract metrics.
    
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
    
    # Get available tags
    scalar_tags = ea.Tags().get("scalars", [])
    print(f"\nAvailable scalar tags ({len(scalar_tags)}):")
    for tag in sorted(scalar_tags):
        print(f"  - {tag}")
    
    # Extract metrics
    metrics = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        if events:
            values = [e.value for e in events]
            steps = [e.step for e in events]
            metrics[tag] = {
                "values": values,
                "steps": steps,
                "min": min(values),
                "max": max(values),
                "latest": values[-1],
                "count": len(values),
            }
    
    return metrics


def print_metrics_summary(metrics: dict) -> None:
    """Print a summary of extracted metrics.
    
    Args:
        metrics: Dictionary of metrics from analyze_tensorboard_logs.
    """
    if not metrics:
        print("No metrics to display.")
        return
    
    print("\n" + "=" * 80)
    print("TRAINING METRICS SUMMARY")
    print("=" * 80)
    
    # Group metrics by category
    train_metrics = {k: v for k, v in metrics.items() if k.startswith("train/")}
    val_metrics = {k: v for k, v in metrics.items() if k.startswith("val/")}
    epoch_metrics = {k: v for k, v in metrics.items() if k.startswith("epoch/")}
    
    if train_metrics:
        print("\n--- Training Metrics (per step) ---")
        for tag in sorted(train_metrics.keys()):
            data = train_metrics[tag]
            print(f"  {tag}:")
            print(f"    Latest: {data['latest']:.6f}")
            print(f"    Range: [{data['min']:.6f}, {data['max']:.6f}]")
            print(f"    Steps: {data['count']} data points")
    
    if val_metrics:
        print("\n--- Validation Metrics (per epoch) ---")
        for tag in sorted(val_metrics.keys()):
            data = val_metrics[tag]
            print(f"  {tag}:")
            print(f"    Latest: {data['latest']:.6f}")
            print(f"    Best: {data['max']:.6f}")
            print(f"    Range: [{data['min']:.6f}, {data['max']:.6f}]")
            print(f"    Epochs: {data['count']} evaluations")
    
    if epoch_metrics:
        print("\n--- Epoch Summary Metrics ---")
        for tag in sorted(epoch_metrics.keys()):
            data = epoch_metrics[tag]
            print(f"  {tag}: {data['latest']:.6f}")
    
    # Highlight key metrics
    print("\n" + "-" * 80)
    print("KEY FINDINGS:")
    print("-" * 80)
    
    if "val/recall@10" in metrics:
        r10_data = metrics["val/recall@10"]
        print("\n  Validation Recall@10:")
        print(f"    Latest: {r10_data['latest']:.4f} ({r10_data['latest']*100:.2f}%)")
        print(f"    Best:   {r10_data['max']:.4f} ({r10_data['max']*100:.2f}%)")
    
    if "val/recall@1" in metrics:
        r1_data = metrics["val/recall@1"]
        print("\n  Validation Recall@1:")
        print(f"    Latest: {r1_data['latest']:.4f} ({r1_data['latest']*100:.2f}%)")
        print(f"    Best:   {r1_data['max']:.4f} ({r1_data['max']*100:.2f}%)")
    
    if "train/loss" in metrics:
        loss_data = metrics["train/loss"]
        print("\n  Training Loss:")
        print(f"    Latest: {loss_data['latest']:.4f}")
        print(f"    Range: [{loss_data['min']:.4f}, {loss_data['max']:.4f}]")
    
    if "epoch/temperature" in metrics:
        temp_data = metrics["epoch/temperature"]
        print(f"\n  Temperature: {temp_data['latest']:.4f}")


def main() -> None:
    """Main entry point."""
    print("VectorMind TensorBoard Log Analysis")
    print("=" * 80)
    
    metrics = analyze_tensorboard_logs()
    print_metrics_summary(metrics)
    
    print("\n" + "=" * 80)
    print("Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
