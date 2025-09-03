#!/usr/bin/env python3
"""
Improved Sentiment Experiment - Using 50 Diverse Examples
This should fix the overfitting issue and create a robust operator
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
print("IMPROVED SENTIMENT EXPERIMENT - 50 DIVERSE EXAMPLES")
print("Fixing overfitting with rich, varied training data")
print("=" * 70)


class ImprovedSentimentCollector:
    """Collect activation data using diverse, creative prompts"""
    
    def __init__(self):
        print("\n[Phase 1: Improved Data Collection]")
        print("Loading GPT-2 with TransformerLens...")
        self.adapter = TransformerLensAdapter("gpt2", "cpu")
        print(f"✓ Model loaded (hidden_dim={self.adapter.get_hidden_dim()})")
        
        # Load diverse prompts
        self.load_diverse_prompts()
        
        # Target layer for extraction
        self.target_layer = 6
        
    def load_diverse_prompts(self):
        """Load the 50 diverse prompts we generated"""
        prompt_file = Path("data/sentiment_experiment/diverse_prompts_50.json")
        
        if not prompt_file.exists():
            raise FileNotFoundError("Diverse prompts not found! Run generate_diverse_prompts.py first.")
        
        with open(prompt_file, 'r') as f:
            data = json.load(f)
        
        self.positive_prompts = data['positive_prompts']
        self.negative_prompts = data['negative_prompts']
        
        print(f"✓ Loaded {len(self.positive_prompts)} diverse positive prompts")
        print(f"✓ Loaded {len(self.negative_prompts)} diverse negative prompts")
        
        # Show a few examples
        print("\nSample positive prompts:")
        for i in range(3):
            print(f"  • {self.positive_prompts[i]}")
        
        print("\nSample negative prompts:")
        for i in range(3):
            print(f"  • {self.negative_prompts[i]}")
    
    def collect_activations(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract activations from all diverse prompts"""
        
        print(f"\nExtracting activations from Layer {self.target_layer}...")
        print("This will take longer but create much better training data!")
        
        positive_activations = []
        negative_activations = []
        
        # Collect positive activations
        print(f"\n🌟 Processing {len(self.positive_prompts)} positive prompts...")
        for i, prompt in enumerate(self.positive_prompts):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(self.positive_prompts)}")
            
            states = self.adapter.extract_hidden_states(prompt, [self.target_layer])
            activation = states[self.target_layer].cpu().numpy()
            positive_activations.append(activation)
        
        # Collect negative activations  
        print(f"\n😞 Processing {len(self.negative_prompts)} negative prompts...")
        for i, prompt in enumerate(self.negative_prompts):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(self.negative_prompts)}")
                
            states = self.adapter.extract_hidden_states(prompt, [self.target_layer])
            activation = states[self.target_layer].cpu().numpy()
            negative_activations.append(activation)
        
        print(f"\n✅ Collected {len(positive_activations)} positive activations")
        print(f"✅ Collected {len(negative_activations)} negative activations")
        
        return positive_activations, negative_activations


class ImprovedHDCEncoder:
    """Enhanced HDC encoder with better statistics"""
    
    def __init__(self, input_dim: int = 768, hd_dim: int = 10000):
        self.input_dim = input_dim
        self.hd_dim = hd_dim
        
        print(f"\n[HDC Encoding - Improved]")
        print(f"Encoder: {input_dim} → {hd_dim} dimensions")
        
        # Create deterministic random projection matrix
        torch.manual_seed(42)
        self.projection = torch.randn(input_dim, hd_dim)
        self.projection = self.projection / torch.norm(self.projection, dim=0, keepdim=True)
        
    def encode_activation(self, activation: np.ndarray) -> np.ndarray:
        """Encode activation to HDC with consistent preprocessing"""
        
        # Convert to tensor
        act_tensor = torch.tensor(activation, dtype=torch.float32)
        
        # Handle different shapes - average over sequence dimension
        if len(act_tensor.shape) == 3:
            act_tensor = act_tensor.mean(dim=[0, 1])  # (batch, seq, dim) -> (dim,)
        elif len(act_tensor.shape) == 2:
            act_tensor = act_tensor.mean(dim=0)       # (seq, dim) -> (dim,)
        
        # Ensure correct dimensionality
        if len(act_tensor.shape) != 1:
            act_tensor = act_tensor.flatten()
        
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
    
    def encode_all_improved(self, positive_acts: List[np.ndarray], 
                           negative_acts: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Encode all activations with improved statistics"""
        
        print(f"\nEncoding {len(positive_acts)} + {len(negative_acts)} activations to cogits...")
        
        positive_cogits = []
        for i, act in enumerate(positive_acts):
            if (i + 1) % 20 == 0:
                print(f"  Positive: {i+1}/{len(positive_acts)}")
            cogit = self.encode_activation(act)
            positive_cogits.append(cogit)
        
        negative_cogits = []
        for i, act in enumerate(negative_acts):
            if (i + 1) % 20 == 0:
                print(f"  Negative: {i+1}/{len(negative_acts)}")
            cogit = self.encode_activation(act)
            negative_cogits.append(cogit)
        
        print(f"\n✅ Encoded {len(positive_cogits)} positive cogits")
        print(f"✅ Encoded {len(negative_cogits)} negative cogits")
        
        # Enhanced separation analysis
        print("\n📊 Improved Separation Analysis:")
        pos_mean = np.mean(positive_cogits, axis=0)
        neg_mean = np.mean(negative_cogits, axis=0)
        
        # Cosine similarity between centroids
        similarity = np.dot(pos_mean, neg_mean) / (np.linalg.norm(pos_mean) * np.linalg.norm(neg_mean))
        print(f"  Centroid similarity: {similarity:.4f}")
        
        # Hamming distance (more interpretable for binary vectors)
        hamming_dist = np.sum(np.sign(pos_mean) != np.sign(neg_mean)) / len(pos_mean)
        print(f"  Centroid Hamming distance: {hamming_dist:.4f} ({hamming_dist*100:.1f}% bits different)")
        
        # Within-class consistency
        pos_consistencies = []
        for cogit in positive_cogits[:10]:  # Sample
            sim = np.dot(cogit, pos_mean) / (np.linalg.norm(cogit) * np.linalg.norm(pos_mean))
            pos_consistencies.append(sim)
        
        print(f"  Positive class consistency: {np.mean(pos_consistencies):.4f} ± {np.std(pos_consistencies):.4f}")
        
        return positive_cogits, negative_cogits


def save_improved_data(positive_cogits: List[np.ndarray], 
                      negative_cogits: List[np.ndarray]) -> Path:
    """Save the improved dataset"""
    
    output_dir = Path("data/sentiment_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save cogits
    data = {
        'positive_cogits': [cogit.tolist() for cogit in positive_cogits],
        'negative_cogits': [cogit.tolist() for cogit in negative_cogits],
        'hd_dim': len(positive_cogits[0]),
        'dataset_size': f"{len(positive_cogits)} positive + {len(negative_cogits)} negative",
        'improvement': 'diverse_prompts_50_examples',
        'timestamp': timestamp
    }
    
    output_file = output_dir / f"improved_cogits_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"\n💾 Saved improved dataset to {output_file}")
    return output_file


def run_improved_phase1():
    """Run Phase 1 with diverse, robust data collection"""
    
    # Collect activations with diverse prompts
    collector = ImprovedSentimentCollector()
    positive_acts, negative_acts = collector.collect_activations()
    
    # Encode to HDC with better statistics
    encoder = ImprovedHDCEncoder()
    positive_cogits, negative_cogits = encoder.encode_all_improved(positive_acts, negative_acts)
    
    # Save improved dataset
    output_file = save_improved_data(positive_cogits, negative_cogits)
    
    print("\n" + "=" * 70)
    print("🎉 IMPROVED PHASE 1 COMPLETE!")
    print("=" * 70)
    
    print(f"\nDataset improvements over original:")
    print(f"• 10 examples → {len(positive_cogits)} examples (5x more data!)")
    print(f"• Simple prompts → Diverse, creative prompts")
    print(f"• Limited vocabulary → Rich emotional language")  
    print(f"• Few sentence types → Questions, statements, exclamations")
    print(f"• Single context → Work, personal, social, sensory contexts")
    
    print(f"\nThis rich dataset should enable the operator to learn:")
    print(f"• General sentiment patterns (not memorized phrases)")
    print(f"• Structure-preserving transformations") 
    print(f"• Robust mappings that generalize to new text")
    
    return output_file, positive_cogits, negative_cogits


if __name__ == "__main__":
    output_file, pos_cogits, neg_cogits = run_improved_phase1()