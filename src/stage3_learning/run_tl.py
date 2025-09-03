#!/usr/bin/env python3
"""
Stage 3: Learn Cognitive Manipulation Operators
Train models to transform cogits along cognitive dimensions.
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import jsonlines
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import yaml

# Deterministic seeding
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class CognitiveOperator(nn.Module):
    """Neural network that learns to transform cogits along a cognitive dimension"""
    
    def __init__(self, hd_dim: int = 10000, hidden_dim: int = 512):
        super().__init__()
        
        # Simple MLP to learn the transformation
        self.transform = nn.Sequential(
            nn.Linear(hd_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hd_dim),
            nn.Tanh()  # Keep outputs bounded
        )
        
    def forward(self, cogit: torch.Tensor) -> torch.Tensor:
        """Apply cognitive transformation"""
        return self.transform(cogit)


def prepare_training_data(cogit_file: Path) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Load cogits and prepare training pairs by dimension"""
    data_by_dimension = {}
    
    with jsonlines.open(cogit_file) as reader:
        for obj in reader:
            dimension = obj['dimension']
            
            if dimension not in data_by_dimension:
                data_by_dimension[dimension] = []
            
            # Convert cogits to tensors
            low_cogit = torch.tensor(obj['low_cogit'], dtype=torch.float32)
            high_cogit = torch.tensor(obj['high_cogit'], dtype=torch.float32)
            
            # Add as training pair (input: low, target: high)
            data_by_dimension[dimension].append((low_cogit, high_cogit))
    
    return data_by_dimension


def train_operator(dimension: str, training_pairs: List[Tuple[torch.Tensor, torch.Tensor]], 
                  epochs: int = 50) -> Tuple[CognitiveOperator, Dict]:
    """Train an operator to transform cogits for a specific dimension"""
    
    print(f"\nTraining operator for '{dimension}' dimension...")
    
    # Initialize model
    model = CognitiveOperator(hd_dim=10000)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training metrics
    losses = []
    
    # Convert to batches
    batch_size = 8
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle data
        shuffled_pairs = random.sample(training_pairs, len(training_pairs))
        
        for i in range(0, len(shuffled_pairs), batch_size):
            batch = shuffled_pairs[i:i+batch_size]
            
            # Prepare batch
            inputs = torch.stack([pair[0] for pair in batch])
            targets = torch.stack([pair[1] for pair in batch])
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Evaluate final performance
    model.eval()
    with torch.no_grad():
        # Test on all pairs
        all_inputs = torch.stack([pair[0] for pair in training_pairs])
        all_targets = torch.stack([pair[1] for pair in training_pairs])
        predictions = model(all_inputs)
        
        # Calculate similarities
        cosine_sim = nn.functional.cosine_similarity(predictions, all_targets, dim=1)
        mean_similarity = cosine_sim.mean().item()
    
    print(f"  ✓ Training complete. Mean output similarity: {mean_similarity:.4f}")
    
    metrics = {
        'dimension': dimension,
        'epochs': epochs,
        'final_loss': losses[-1],
        'mean_similarity': mean_similarity,
        'training_samples': len(training_pairs)
    }
    
    return model, metrics


def run_learning_pipeline():
    """Run the full operator learning pipeline"""
    print("\n" + "=" * 60)
    print("STAGE 3: OPERATOR LEARNING")
    print("=" * 60)
    
    # Find the latest cogit file
    cogit_dir = Path("data/processed/cogits")
    cogit_files = list(cogit_dir.glob("cogits_*.jsonl"))
    
    if not cogit_files:
        print("No cogit files found! Run Stage 2 first.")
        return None
    
    # Use the most recent file
    latest_file = max(cogit_files, key=lambda p: p.stat().st_mtime)
    print(f"\nUsing cogit file: {latest_file}")
    
    # Prepare training data
    data_by_dimension = prepare_training_data(latest_file)
    print(f"✓ Loaded training data for dimensions: {', '.join(data_by_dimension.keys())}")
    
    # Train operators for each dimension
    operators = {}
    all_metrics = {}
    
    for dimension, pairs in data_by_dimension.items():
        print(f"\n{'='*40}")
        model, metrics = train_operator(dimension, pairs)
        operators[dimension] = model
        all_metrics[dimension] = metrics
    
    # Save models and metrics
    output_dir = Path("models/operators")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for dimension, model in operators.items():
        model_file = output_dir / f"operator_{dimension}_{timestamp}.pt"
        torch.save(model.state_dict(), model_file)
        print(f"\n✓ Saved {dimension} operator to {model_file}")
    
    # Save metrics
    metrics_file = Path("results/metrics") / f"training_metrics_{timestamp}.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"✓ Saved metrics to {metrics_file}")
    
    return operators, all_metrics


def test_operators(operators: Dict[str, CognitiveOperator]):
    """Test the learned operators"""
    print("\n" + "=" * 60)
    print("TESTING LEARNED OPERATORS")
    print("=" * 60)
    
    # Create a random test cogit
    test_cogit = torch.sign(torch.randn(10000))
    
    for dimension, model in operators.items():
        model.eval()
        with torch.no_grad():
            # Apply operator
            transformed = model(test_cogit.unsqueeze(0))
            
            # Check properties
            similarity = nn.functional.cosine_similarity(
                test_cogit.unsqueeze(0), 
                transformed, 
                dim=1
            ).item()
            
            print(f"\n{dimension.upper()} operator:")
            print(f"  Input norm:  {torch.norm(test_cogit).item():.2f}")
            print(f"  Output norm: {torch.norm(transformed).item():.2f}")
            print(f"  Similarity:  {similarity:.4f}")


if __name__ == "__main__":
    # Run the learning pipeline
    operators, metrics = run_learning_pipeline()
    
    if operators:
        # Test the operators
        test_operators(operators)
        
        print("\n" + "=" * 60)
        print("Stage 3 Complete - Operators Trained Successfully!")
        print("=" * 60)
        print("\nPipeline Summary:")
        print("1. ✓ Extracted activations with TransformerLens (no deadlocks!)")
        print("2. ✓ Encoded into hyperdimensional cogits")
        print("3. ✓ Learned cognitive manipulation operators")
        print("\nThe system is now ready for cognitive manipulation experiments!")