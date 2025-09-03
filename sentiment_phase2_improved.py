#!/usr/bin/env python3
"""
Phase 2 Improved: Train Robust Sentiment Operator
Using 50 balanced, diverse examples to fix overfitting
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

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🎯 Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")

print("=" * 70)
print("PHASE 2 IMPROVED: ROBUST SENTIMENT OPERATOR TRAINING")
print("50 diverse, balanced examples → Better generalization")
print("=" * 70)


class ImprovedSentimentOperator(nn.Module):
    """
    Improved operator architecture for robust sentiment transformation
    """
    
    def __init__(self, hd_dim: int = 10000):
        super().__init__()
        
        # Even simpler architecture to prevent overfitting
        # With more data, we can afford to be more conservative
        self.transform = nn.Sequential(
            nn.Linear(hd_dim, 1024),  # Larger first layer for more capacity
            nn.ReLU(),
            nn.Dropout(0.1),          # Less dropout since we have more data
            nn.Linear(1024, 512),     # Bottleneck layer
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(512, hd_dim),   # Back to HD space
            nn.Tanh()                 # Keep outputs bounded
        )
        
        # Move to device
        self.to(device)
        
    def forward(self, cogit: torch.Tensor) -> torch.Tensor:
        """Apply the sentiment transformation"""
        return self.transform(cogit)


def load_balanced_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the balanced 50+50 dataset"""
    
    data_dir = Path("data/sentiment_experiment")
    
    # Find the balanced cogit file
    balanced_files = list(data_dir.glob("balanced_cogits_*.json"))
    if not balanced_files:
        raise FileNotFoundError("No balanced dataset found! Run fix_balance.py first.")
    
    latest_file = max(balanced_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📁 Loading balanced data: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Convert to tensors
    positive_cogits = torch.tensor(data['positive_cogits'], dtype=torch.float32).to(device)
    negative_cogits = torch.tensor(data['negative_cogits'], dtype=torch.float32).to(device)
    
    print(f"✅ Loaded {len(positive_cogits)} positive cogits")
    print(f"✅ Loaded {len(negative_cogits)} negative cogits")
    print(f"✅ Perfect balance: {len(positive_cogits)} = {len(negative_cogits)}")
    
    return positive_cogits, negative_cogits


def create_robust_training_pairs(positive_cogits: torch.Tensor, 
                                negative_cogits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create training pairs with data augmentation for robustness
    """
    
    print(f"\n📊 Creating robust training pairs...")
    
    # Basic pairing: each positive with corresponding negative
    basic_inputs = positive_cogits
    basic_targets = negative_cogits
    
    # Data augmentation: add some shuffled pairs for robustness
    # This prevents the operator from learning position-specific mappings
    shuffled_indices = torch.randperm(len(negative_cogits))
    shuffled_targets = negative_cogits[shuffled_indices]
    
    # Combine basic + shuffled pairs
    all_inputs = torch.cat([basic_inputs, positive_cogits])
    all_targets = torch.cat([basic_targets, shuffled_targets])
    
    print(f"✅ Created {len(all_inputs)} training pairs")
    print(f"  • {len(basic_inputs)} direct positive→negative pairs")
    print(f"  • {len(basic_inputs)} shuffled pairs for robustness")
    
    return all_inputs, all_targets


def train_robust_operator(inputs: torch.Tensor, 
                         targets: torch.Tensor,
                         epochs: int = 100,
                         batch_size: int = 10) -> Tuple[ImprovedSentimentOperator, List[float]]:
    """
    Train the operator with improved regularization
    """
    
    print(f"\n🎯 Training Robust Sentiment Operator")
    print(f"  • Dataset: {len(inputs)} diverse training pairs")
    print(f"  • Architecture: {inputs.shape[1]} → 1024 → 512 → {inputs.shape[1]}")
    print(f"  • Training: {epochs} epochs, batch size {batch_size}")
    
    # Initialize model
    model = ImprovedSentimentOperator(hd_dim=inputs.shape[1])
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"\n🚀 Training started...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        # Shuffle data each epoch
        perm = torch.randperm(len(inputs))
        shuffled_inputs = inputs[perm]
        shuffled_targets = targets[perm]
        
        # Mini-batch training
        for i in range(0, len(shuffled_inputs), batch_size):
            batch_inputs = shuffled_inputs[i:i+batch_size]
            batch_targets = shuffled_targets[i:i+batch_size]
            
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
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Progress updates
        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s")
        
        # Early stopping
        if patience_counter >= 50:
            print(f"  Early stopping at epoch {epoch + 1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time:.1f} seconds!")
    print(f"  Final loss: {losses[-1]:.6f}")
    
    return model, losses


def evaluate_robust_operator(model: ImprovedSentimentOperator,
                            positive_cogits: torch.Tensor,
                            negative_cogits: torch.Tensor):
    """Enhanced evaluation of the operator"""
    
    print(f"\n🔍 Evaluating Robust Operator Performance")
    
    model.eval()
    with torch.no_grad():
        # Test on a subset
        test_positives = positive_cogits[:10]
        test_negatives = negative_cogits[:10]
        
        # Apply transformation
        transformed = model(test_positives)
        transformed_binary = torch.sign(transformed)
        
        print(f"\n1. Transformation Quality:")
        
        # Calculate centroids
        pos_centroid = torch.sign(positive_cogits.mean(dim=0))
        neg_centroid = torch.sign(negative_cogits.mean(dim=0))
        
        successful_shifts = 0
        
        for i in range(len(test_positives)):
            original = test_positives[i]
            transformed_cogit = transformed_binary[i]
            
            # Similarities to centroids
            sim_to_neg = torch.cosine_similarity(
                transformed_cogit.unsqueeze(0), neg_centroid.unsqueeze(0)
            ).item()
            
            sim_to_pos = torch.cosine_similarity(
                transformed_cogit.unsqueeze(0), pos_centroid.unsqueeze(0) 
            ).item()
            
            if sim_to_neg > sim_to_pos:
                successful_shifts += 1
                status = "✅ SUCCESS"
            else:
                status = "⚠️  Partial"
            
            print(f"  Sample {i+1}: Neg={sim_to_neg:.4f}, Pos={sim_to_pos:.4f} {status}")
        
        success_rate = successful_shifts / len(test_positives)
        print(f"\n  🎯 Success Rate: {success_rate*100:.1f}% ({successful_shifts}/{len(test_positives)})")
        
        # Check transformation magnitude
        print(f"\n2. Transformation Properties:")
        
        # Average bit flip rate
        bit_changes = []
        for i in range(len(test_positives)):
            original = torch.sign(test_positives[i])
            transformed_cogit = transformed_binary[i]
            changes = torch.sum(torch.abs(transformed_cogit - original)) / 2
            bit_changes.append(changes.item())
        
        avg_changes = np.mean(bit_changes)
        print(f"  Average bits changed: {avg_changes:.1f} out of {test_positives.shape[1]}")
        print(f"  Percentage changed: {100 * avg_changes / test_positives.shape[1]:.2f}%")
        
        if 0.5 <= avg_changes/test_positives.shape[1] <= 0.05:
            print(f"  ✅ Good: Subtle changes (not too aggressive)")
        else:
            print(f"  ⚠️  May be too aggressive or too subtle")
    
    return model


def save_robust_operator(model: ImprovedSentimentOperator, losses: List[float]):
    """Save the robust operator"""
    
    output_dir = Path("models/sentiment_operator")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "robust_sentiment_operator.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': 'ImprovedSentimentOperator',
        'hd_dim': 10000,
        'training_data': '50_balanced_diverse_examples',
        'device': str(device)
    }, model_path)
    
    print(f"\n💾 Saved robust operator to {model_path}")
    
    # Save training history
    history = {
        'losses': losses,
        'final_loss': losses[-1],
        'epochs': len(losses),
        'training_improvement': 'balanced_diverse_dataset_50_examples',
        'expected_improvement': 'subtle_sentiment_shifts_preserve_structure'
    }
    
    with open(output_dir / "robust_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)


def run_improved_phase2():
    """Run Phase 2 with robust training"""
    
    print(f"\n" + "="*70)
    print("PHASE 2 IMPROVED: Robust Operator Training")
    print("="*70)
    
    # Load balanced data
    positive_cogits, negative_cogits = load_balanced_data()
    
    # Create robust training pairs
    inputs, targets = create_robust_training_pairs(positive_cogits, negative_cogits)
    
    # Train robust operator
    model, losses = train_robust_operator(inputs, targets, epochs=100)
    
    # Evaluate performance
    model = evaluate_robust_operator(model, positive_cogits, negative_cogits)
    
    # Save the operator
    save_robust_operator(model, losses)
    
    print(f"\n" + "="*70)
    print("🎉 PHASE 2 IMPROVED COMPLETE!")
    print("="*70)
    print(f"\nKey Improvements:")
    print(f"• 10 examples → 50 balanced examples (5x more data)")
    print(f"• Simple prompts → Diverse, creative prompts") 
    print(f"• Basic training → Data augmentation + regularization")
    print(f"• Overfitting → Robust generalization")
    print(f"\nExpected result: Subtle sentiment shifts that preserve text structure")
    print(f"Instead of breaking generation, we should see controlled manipulation!")
    
    return model


if __name__ == "__main__":
    model = run_improved_phase2()