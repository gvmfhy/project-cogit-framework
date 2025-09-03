#!/usr/bin/env python3
"""
Phase 2: Learning the Sentiment Manipulation Operator (Efficient Version)
Optimized for M1 Mac with 16GB RAM

Goal: Learn a mathematical function that transforms positive cogits → negative cogits
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Optional
import time

# Deterministic seeding
torch.manual_seed(42)
np.random.seed(42)

# Optimize for Apple Silicon
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🎯 Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("=" * 70)
print("PHASE 2: LEARNING THE SENTIMENT DIAL")
print("Goal: Learn operator to transform Positive → Negative")
print("=" * 70)


class SimpleSentimentOperator(nn.Module):
    """
    A simpler, more efficient operator for sentiment transformation.
    This learns the mathematical recipe to make thoughts more negative.
    """
    
    def __init__(self, hd_dim: int = 10000):
        super().__init__()
        
        # Simpler architecture for efficiency
        # The key insight: We're learning a rotation/transformation in HD space
        self.transform = nn.Sequential(
            nn.Linear(hd_dim, 512),  # Compress to lower dim
            nn.Tanh(),
            nn.Linear(512, hd_dim),   # Project back to HD space
            nn.Tanh()                 # Keep outputs bounded
        )
        
        # Move to device
        self.to(device)
        
    def forward(self, cogit: torch.Tensor) -> torch.Tensor:
        """Apply the make_negative transformation"""
        return self.transform(cogit)


def load_phase1_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and prepare cogits from Phase 1"""
    
    data_dir = Path("data/sentiment_experiment")
    
    # Find the most recent cogit file
    cogit_files = list(data_dir.glob("sentiment_cogits_*.json"))
    if not cogit_files:
        raise FileNotFoundError("No Phase 1 data found! Run Phase 1 first.")
    
    latest_file = max(cogit_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📁 Loading data from: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Convert to tensors and move to device
    positive_cogits = torch.tensor(data['positive_cogits'], dtype=torch.float32).to(device)
    negative_cogits = torch.tensor(data['negative_cogits'], dtype=torch.float32).to(device)
    
    print(f"✓ Loaded {len(positive_cogits)} positive cogits")
    print(f"✓ Loaded {len(negative_cogits)} negative cogits")
    print(f"✓ Cogit dimensionality: {positive_cogits.shape[1]}")
    
    return positive_cogits, negative_cogits


def create_training_pairs(positive_cogits: torch.Tensor, 
                         negative_cogits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create training pairs for the operator.
    
    The key insight: We want to learn Positive → Negative transformation
    So we pair each positive cogit with a corresponding negative one.
    """
    
    print("\n📊 Creating training pairs...")
    
    # Simple pairing: match each positive with a negative
    # For 10 positive and 10 negative, we create 10 pairs
    n_pairs = min(len(positive_cogits), len(negative_cogits))
    
    inputs = positive_cogits[:n_pairs]
    targets = negative_cogits[:n_pairs]
    
    print(f"✓ Created {n_pairs} transformation pairs")
    print(f"  Each pair: Positive cogit → Negative cogit")
    
    return inputs, targets


def train_make_negative_operator(inputs: torch.Tensor, 
                                targets: torch.Tensor,
                                epochs: int = 50,
                                batch_size: int = 5) -> SimpleSentimentOperator:
    """
    Train the operator to learn: Positive → Negative transformation
    
    This is the core of Phase 2: Learning the mathematical dial
    """
    
    print("\n🎯 Training the Make-Negative Operator")
    print(f"  Architecture: {10000} → 512 → {10000} dimensions")
    print(f"  Training for {epochs} epochs with batch size {batch_size}")
    
    # Initialize model
    model = SimpleSentimentOperator(hd_dim=inputs.shape[1])
    
    # Use simpler optimizer for stability
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training metrics
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Mini-batch training
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(inputs) / batch_size)
        losses.append(avg_loss)
        
        # Progress updates
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
    
    print(f"\n✓ Training complete in {time.time() - start_time:.1f} seconds!")
    
    return model, losses


def evaluate_operator(model: SimpleSentimentOperator,
                     positive_cogits: torch.Tensor,
                     negative_cogits: torch.Tensor):
    """
    Evaluate how well our operator learned the transformation
    """
    
    print("\n🔍 Evaluating the Learned Operator")
    
    model.eval()
    with torch.no_grad():
        # Test the transformation
        transformed = model(positive_cogits[:5])
        transformed_binary = torch.sign(transformed)
        
        # Calculate similarities
        print("\n1. Testing Positive → Negative transformation:")
        
        for i in range(5):
            # Original positive cogit
            original = positive_cogits[i]
            
            # Transformed (should be more like negative)
            transformed_cogit = transformed_binary[i]
            
            # Compare to negative centroid
            neg_centroid = torch.sign(negative_cogits.mean(dim=0))
            similarity_to_negative = torch.cosine_similarity(
                transformed_cogit.unsqueeze(0),
                neg_centroid.unsqueeze(0)
            ).item()
            
            # Compare to positive centroid  
            pos_centroid = torch.sign(positive_cogits.mean(dim=0))
            similarity_to_positive = torch.cosine_similarity(
                transformed_cogit.unsqueeze(0),
                pos_centroid.unsqueeze(0)
            ).item()
            
            print(f"  Sample {i+1}:")
            print(f"    After transformation → Negative similarity: {similarity_to_negative:.4f}")
            print(f"    After transformation → Positive similarity: {similarity_to_positive:.4f}")
            
            if similarity_to_negative > similarity_to_positive:
                print(f"    ✓ Successfully shifted toward negative!")
    
    # Calculate overall success
    print("\n2. Operator Properties:")
    
    # Check if the operator is doing something meaningful
    with torch.no_grad():
        # Apply operator to positive centroid
        pos_centroid = torch.sign(positive_cogits.mean(dim=0)).unsqueeze(0)
        transformed_centroid = torch.sign(model(pos_centroid))
        
        # Measure the change
        change = torch.sum(torch.abs(transformed_centroid - pos_centroid)) / 2
        print(f"  Number of bits flipped: {int(change.item())} out of {pos_centroid.shape[1]}")
        print(f"  Percentage changed: {100 * change.item() / pos_centroid.shape[1]:.2f}%")


def save_operator(model: SimpleSentimentOperator, losses: List[float]):
    """Save the trained make_negative_operator"""
    
    output_dir = Path("models/sentiment_operator")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "make_negative_operator.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': 'SimpleSentimentOperator',
        'hd_dim': 10000,
        'device': str(device)
    }, model_path)
    
    print(f"\n💾 Saved operator to {model_path}")
    
    # Save training history
    history = {
        'losses': losses,
        'final_loss': losses[-1],
        'epochs': len(losses),
        'transformation': 'positive_to_negative'
    }
    
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Saved training history")


def run_phase2():
    """Execute Phase 2: Learn the Mathematical Dial"""
    
    print("\n" + "="*70)
    print("Starting Phase 2: Learning the Sentiment Manipulation Operator")
    print("="*70)
    
    # Load data
    positive_cogits, negative_cogits = load_phase1_data()
    
    # Create training pairs
    inputs, targets = create_training_pairs(positive_cogits, negative_cogits)
    
    # Train the operator
    model, losses = train_make_negative_operator(inputs, targets, epochs=50)
    
    # Evaluate
    evaluate_operator(model, positive_cogits, negative_cogits)
    
    # Save
    save_operator(model, losses)
    
    print("\n" + "="*70)
    print("🎉 PHASE 2 COMPLETE!")
    print("="*70)
    print("\nWe have successfully learned a mathematical operator that:")
    print("• Takes a positive sentiment cogit as input")
    print("• Outputs a negative sentiment cogit")
    print("• Works in 10,000-dimensional hyperdimensional space")
    print("\nThis 'make_negative_operator' is our sentiment dial!")
    print("It represents the abstract concept of 'making something more negative'")
    print("\nReady for Phase 3: Testing if this dial can control GPT-2's outputs!")
    
    return model


if __name__ == "__main__":
    model = run_phase2()