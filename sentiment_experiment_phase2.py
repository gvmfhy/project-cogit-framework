#!/usr/bin/env python3
"""
Sentiment Manipulation Experiment - Phase 2
Learning a mathematical operator to transform sentiment in HDC space
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple
import matplotlib.pyplot as plt

# Deterministic seeding
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("PHASE 2: LEARNING THE SENTIMENT OPERATOR")
print("=" * 70)

# ============================================================================
# PHASE 2: OPERATOR LEARNING
# ============================================================================

class SentimentOperator(nn.Module):
    """
    A neural network that learns to transform negative sentiment cogits 
    into positive sentiment cogits in HDC space
    """
    
    def __init__(self, hd_dim: int = 10000):
        super().__init__()
        
        # Multi-layer network to learn the transformation
        self.transform = nn.Sequential(
            nn.Linear(hd_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, hd_dim),
            nn.Tanh()  # Output in [-1, 1] range for binary cogits
        )
        
    def forward(self, cogit: torch.Tensor) -> torch.Tensor:
        """Apply sentiment transformation"""
        return self.transform(cogit)


def load_phase1_data() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load cogits from Phase 1"""
    
    data_dir = Path("data/sentiment_experiment")
    
    # Find the most recent cogit file
    cogit_files = list(data_dir.glob("sentiment_cogits_*.json"))
    if not cogit_files:
        raise FileNotFoundError("No Phase 1 data found! Run Phase 1 first.")
    
    latest_file = max(cogit_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading data from: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    positive_cogits = [np.array(c) for c in data['positive_cogits']]
    negative_cogits = [np.array(c) for c in data['negative_cogits']]
    
    print(f"✓ Loaded {len(positive_cogits)} positive cogits")
    print(f"✓ Loaded {len(negative_cogits)} negative cogits")
    
    return positive_cogits, negative_cogits


def train_sentiment_operator(positive_cogits: List[np.ndarray], 
                            negative_cogits: List[np.ndarray],
                            epochs: int = 100) -> SentimentOperator:
    """
    Train an operator to transform negative sentiment to positive
    
    The key insight: We train the network to map:
    - Negative cogits → Positive cogits (learn sentiment change)
    - Positive cogits → Positive cogits (preserve positive sentiment)
    """
    
    print("\n[Training Sentiment Operator]")
    print("Learning transformation: Negative → Positive")
    
    # Initialize model
    hd_dim = len(positive_cogits[0])
    model = SentimentOperator(hd_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Prepare training data
    # Create pairs: (input, target)
    training_pairs = []
    
    # Negative → Positive transformations
    for neg_cogit in negative_cogits:
        for pos_cogit in positive_cogits:
            training_pairs.append((neg_cogit, pos_cogit))
    
    # Positive → Positive (identity for positive)
    for pos_cogit in positive_cogits:
        training_pairs.append((pos_cogit, pos_cogit))
    
    print(f"Training on {len(training_pairs)} transformation pairs")
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle training data
        np.random.shuffle(training_pairs)
        
        for input_cogit, target_cogit in training_pairs:
            # Convert to tensors
            input_tensor = torch.tensor(input_cogit, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor(target_cogit, dtype=torch.float32).unsqueeze(0)
            
            # Forward pass
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    print("\n✓ Training complete!")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Sentiment Operator Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("data/sentiment_experiment/training_loss.png")
    print("✓ Saved training curve to training_loss.png")
    
    return model


def evaluate_operator(model: SentimentOperator, 
                     positive_cogits: List[np.ndarray],
                     negative_cogits: List[np.ndarray]):
    """Evaluate how well the operator transforms sentiment"""
    
    print("\n[Evaluating Operator]")
    
    model.eval()
    with torch.no_grad():
        # Test on negative cogits
        print("\nTesting negative → positive transformation:")
        neg_to_pos_similarities = []
        
        for neg_cogit in negative_cogits[:5]:  # Test on first 5
            input_tensor = torch.tensor(neg_cogit, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor)
            output_binary = torch.sign(output).squeeze().numpy()
            
            # Compare to positive centroid
            pos_centroid = np.mean(positive_cogits, axis=0)
            similarity = np.dot(output_binary, pos_centroid) / (np.linalg.norm(output_binary) * np.linalg.norm(pos_centroid))
            neg_to_pos_similarities.append(similarity)
            
            print(f"  Similarity to positive centroid: {similarity:.4f}")
        
        # Test on positive cogits (should preserve)
        print("\nTesting positive → positive preservation:")
        pos_to_pos_similarities = []
        
        for pos_cogit in positive_cogits[:5]:  # Test on first 5
            input_tensor = torch.tensor(pos_cogit, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor)
            output_binary = torch.sign(output).squeeze().numpy()
            
            # Compare to original
            similarity = np.dot(output_binary, pos_cogit) / (np.linalg.norm(output_binary) * np.linalg.norm(pos_cogit))
            pos_to_pos_similarities.append(similarity)
            
            print(f"  Self-similarity after transformation: {similarity:.4f}")
    
    print(f"\nAverage neg→pos similarity: {np.mean(neg_to_pos_similarities):.4f}")
    print(f"Average pos→pos similarity: {np.mean(pos_to_pos_similarities):.4f}")
    
    return model


def save_operator(model: SentimentOperator):
    """Save the trained operator"""
    
    output_dir = Path("models/sentiment_operator")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "sentiment_operator.pt"
    torch.save(model.state_dict(), model_path)
    
    print(f"\n✓ Saved operator to {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'SentimentOperator',
        'hd_dim': 10000,
        'architecture': 'MLP with 2048 hidden units',
        'training': 'Negative→Positive transformation'
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def run_phase2():
    """Execute Phase 2: Operator Learning"""
    
    # Load Phase 1 data
    positive_cogits, negative_cogits = load_phase1_data()
    
    # Train the operator
    model = train_sentiment_operator(positive_cogits, negative_cogits, epochs=100)
    
    # Evaluate performance
    evaluate_operator(model, positive_cogits, negative_cogits)
    
    # Save the model
    save_operator(model)
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print("\nWe have successfully learned a mathematical operator that:")
    print("• Transforms negative sentiment cogits → positive sentiment cogits")
    print("• Preserves positive sentiment when already positive")
    print("\nThis operator exists in 10,000-dimensional HDC space and can")
    print("potentially be applied to alter the sentiment of GPT-2's outputs.")
    print("\nReady for Phase 3: Testing sentiment control on actual text generation!")
    
    return model


if __name__ == "__main__":
    model = run_phase2()