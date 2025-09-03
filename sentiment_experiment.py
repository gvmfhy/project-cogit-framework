#!/usr/bin/env python3
"""
Sentiment Manipulation Experiment
Question: Can we learn a mathematical operator within HDC to predictably alter sentiment?

Phase 1: Data Collection - Gathering the "Mind States"
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict, Tuple
from src.model_adapter_tl import TransformerLensAdapter

# Deterministic seeding
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("SENTIMENT MANIPULATION EXPERIMENT")
print("Can we control sentiment through HDC operators?")
print("=" * 70)

# ============================================================================
# PHASE 1: DATA COLLECTION
# ============================================================================

class SentimentDataCollector:
    """Collects activation data for positive and negative sentiment"""
    
    def __init__(self):
        # Initialize TransformerLens adapter
        print("\n[Phase 1: Data Collection]")
        print("Initializing GPT-2 with TransformerLens...")
        self.adapter = TransformerLensAdapter("gpt2", "cpu")
        print(f"✓ Model loaded (hidden_dim={self.adapter.get_hidden_dim()})")
        
        # Define paired prompts
        self.positive_prompts = [
            "I love my new puppy, he is so",
            "That was a wonderful and happy",
            "The weather today is absolutely beautiful and",
            "I'm feeling great because",
            "This is the best day ever since",
            "Everything worked out perfectly and I'm",
            "The food was delicious and",
            "I'm so grateful for",
            "Life is amazing when",
            "I'm excited about tomorrow because"
        ]
        
        self.negative_prompts = [
            "The traffic was horrible this morning, it was",
            "I had a terrible and awful",
            "The weather today is absolutely miserable and",
            "I'm feeling terrible because",
            "This is the worst day ever since",
            "Everything went wrong and I'm",
            "The food was disgusting and",
            "I'm so frustrated about",
            "Life is difficult when",
            "I'm worried about tomorrow because"
        ]
        
        # Target layer for extraction (middle layer)
        self.target_layer = 6
        
    def collect_activations(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract activations for positive and negative prompts"""
        
        print(f"\nExtracting activations from Layer {self.target_layer}...")
        
        positive_activations = []
        negative_activations = []
        
        # Collect positive activations
        print("\nPositive prompts:")
        for i, prompt in enumerate(self.positive_prompts):
            states = self.adapter.extract_hidden_states(prompt, [self.target_layer])
            activation = states[self.target_layer].cpu().numpy()
            positive_activations.append(activation)
            print(f"  [{i+1}/10] '{prompt[:30]}...' → shape {activation.shape}")
        
        # Collect negative activations  
        print("\nNegative prompts:")
        for i, prompt in enumerate(self.negative_prompts):
            states = self.adapter.extract_hidden_states(prompt, [self.target_layer])
            activation = states[self.target_layer].cpu().numpy()
            negative_activations.append(activation)
            print(f"  [{i+1}/10] '{prompt[:30]}...' → shape {activation.shape}")
        
        print(f"\n✓ Collected {len(positive_activations)} positive activations")
        print(f"✓ Collected {len(negative_activations)} negative activations")
        
        return positive_activations, negative_activations


class HDCEncoder:
    """Encodes activations into hyperdimensional cognitive vectors"""
    
    def __init__(self, input_dim: int = 768, hd_dim: int = 10000):
        self.input_dim = input_dim
        self.hd_dim = hd_dim
        
        print(f"\n[HDC Encoding]")
        print(f"Initializing HDC encoder ({input_dim} → {hd_dim} dimensions)")
        
        # Create deterministic random projection matrix
        torch.manual_seed(42)
        self.projection = torch.randn(input_dim, hd_dim)
        self.projection = self.projection / torch.norm(self.projection, dim=0, keepdim=True)
        
    def encode_activation(self, activation: np.ndarray) -> np.ndarray:
        """Encode an activation into a hyperdimensional cogit"""
        
        # Convert to tensor
        act_tensor = torch.tensor(activation, dtype=torch.float32)
        
        # Handle different shapes - we expect (batch, seq, dim)
        if len(act_tensor.shape) == 3:
            # Average over batch and sequence dimensions
            act_tensor = act_tensor.mean(dim=[0, 1])
        elif len(act_tensor.shape) == 2:
            # Average over sequence dimension
            act_tensor = act_tensor.mean(dim=0)
        
        # Now act_tensor should be 1D with shape (768,)
        if len(act_tensor.shape) != 1:
            act_tensor = act_tensor.flatten()
        
        # Ensure correct dimensionality
        if act_tensor.shape[0] > self.input_dim:
            act_tensor = act_tensor[:self.input_dim]
        elif act_tensor.shape[0] < self.input_dim:
            padding = torch.zeros(self.input_dim - act_tensor.shape[0])
            act_tensor = torch.cat([act_tensor, padding])
        
        # Project to HD space
        hd_vector = torch.matmul(act_tensor, self.projection)
        
        # Binarize to create cogit
        cogit = torch.sign(hd_vector)
        
        return cogit.numpy()
    
    def encode_all(self, positive_acts: List[np.ndarray], 
                   negative_acts: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Encode all activations into cogits"""
        
        print("\nEncoding activations to HDC cogits...")
        
        positive_cogits = []
        for act in positive_acts:
            cogit = self.encode_activation(act)
            positive_cogits.append(cogit)
        
        negative_cogits = []
        for act in negative_acts:
            cogit = self.encode_activation(act)
            negative_cogits.append(cogit)
        
        print(f"✓ Encoded {len(positive_cogits)} positive cogits")
        print(f"✓ Encoded {len(negative_cogits)} negative cogits")
        
        # Check separation
        pos_mean = np.mean(positive_cogits, axis=0)
        neg_mean = np.mean(negative_cogits, axis=0)
        
        # Cosine similarity between centroids
        similarity = np.dot(pos_mean, neg_mean) / (np.linalg.norm(pos_mean) * np.linalg.norm(neg_mean))
        print(f"\nCentroid similarity: {similarity:.4f}")
        print("(Lower similarity = better separation)")
        
        return positive_cogits, negative_cogits


def save_phase1_data(positive_cogits: List[np.ndarray], 
                     negative_cogits: List[np.ndarray]) -> Path:
    """Save Phase 1 data for later phases"""
    
    output_dir = Path("data/sentiment_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save cogits
    data = {
        'positive_cogits': [cogit.tolist() for cogit in positive_cogits],
        'negative_cogits': [cogit.tolist() for cogit in negative_cogits],
        'hd_dim': len(positive_cogits[0]),
        'timestamp': timestamp
    }
    
    output_file = output_dir / f"sentiment_cogits_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"\n✓ Saved Phase 1 data to {output_file}")
    return output_file


def run_phase1():
    """Execute Phase 1: Data Collection"""
    
    # Collect activations
    collector = SentimentDataCollector()
    positive_acts, negative_acts = collector.collect_activations()
    
    # Encode to HDC
    encoder = HDCEncoder()
    positive_cogits, negative_cogits = encoder.encode_all(positive_acts, negative_acts)
    
    # Save data
    output_file = save_phase1_data(positive_cogits, negative_cogits)
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"\nWe now have:")
    print(f"• {len(positive_cogits)} positive sentiment cogits")
    print(f"• {len(negative_cogits)} negative sentiment cogits")
    print(f"• Each cogit is a {len(positive_cogits[0])}-dimensional binary vector")
    print("\nThese represent the 'mind states' for positive and negative sentiment.")
    print("Ready for Phase 2: Learning the manipulation operator!")
    
    return output_file, positive_cogits, negative_cogits


if __name__ == "__main__":
    output_file, pos_cogits, neg_cogits = run_phase1()