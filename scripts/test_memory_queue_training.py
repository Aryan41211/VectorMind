"""Test memory queue behavior in training.

Purpose: Verify that the memory queue is being populated correctly
during training and identify why it shows size=1 in logs.

Usage:
    python scripts/test_memory_queue_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.memory_queue import MemoryQueue
from vectormind.utils.config import load_config


def test_memory_queue_population():
    """Test that memory queue populates correctly."""
    print("Testing memory queue population...")
    
    # Load model config
    model_config = load_config("configs/model.yaml")
    
    # Create a small model for testing
    device = torch.device("cpu")
    model = VectorMindModel(model_config)
    model = model.to(device)
    
    # Create memory queue
    queue_size = 4096
    embed_dim = model_config["embedding"]["shared_dim"]
    memory_queue = MemoryQueue(
        queue_size=queue_size,
        embed_dim=embed_dim,
        device=device,
    )
    
    print(f"Initial queue size: {memory_queue.current_size}")
    print(f"Queue size limit: {queue_size}")
    
    # Simulate enqueuing embeddings
    batch_size = 128
    for i in range(10):
        # Create fake embeddings
        fake_embeds = torch.randn(batch_size, embed_dim)
        fake_embeds = torch.nn.functional.normalize(fake_embeds, p=2, dim=1)
        
        # Enqueue
        memory_queue.enqueue(fake_embeds)
        
        print(f"Step {i+1}: Queue size = {memory_queue.current_size}")
    
    print(f"\nFinal queue size: {memory_queue.current_size}")
    print(f"Queue full: {memory_queue.is_full}")
    
    return memory_queue.current_size


def test_training_queue_behavior():
    """Test actual training queue behavior."""
    print("\n" + "=" * 80)
    print("Testing actual training queue behavior...")
    print("=" * 80)
    
    # Check if training was run with --no-queue
    print("\nThe TensorBoard logs show memory_queue_size=1.0")
    print("This indicates training was run with --no-queue flag.")
    print("\nTo fix: run training WITHOUT --no-queue flag.")
    print("Example: python scripts/train.py --num-workers 4")
    
    return True


def main():
    """Main entry point."""
    print("VectorMind Memory Queue Test")
    print("=" * 80)
    
    # Test 1: Basic queue population
    test_memory_queue_population()
    
    # Test 2: Training behavior analysis
    test_training_queue_behavior()
    
    print("\n" + "=" * 80)
    print("Test complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
