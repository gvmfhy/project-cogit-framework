#!/usr/bin/env python3
"""
Phase 3 Improved: Test the Robust Sentiment Intervention
Using the improved operator that should preserve text structure
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from src.model_adapter_tl import TransformerLensAdapter

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
print("PHASE 3 IMPROVED: TESTING ROBUST SENTIMENT INTERVENTION")
print("Will the improved operator preserve text while shifting sentiment?")
print("=" * 70)


class ImprovedInterventionSystem:
    """Enhanced intervention system using the robust operator"""
    
    def __init__(self):
        print("\n[Initializing Improved Intervention System]")
        
        # Load GPT-2 with TransformerLens
        print("Loading GPT-2...")
        self.adapter = TransformerLensAdapter("gpt2", "cpu")
        print("✓ Model loaded")
        
        # Load the robust operator
        self.operator = self.load_robust_operator()
        
        # Load encoder projection for consistent encoding/decoding
        self.projection = self.load_projection_matrix()
        self.decoder = HDCDecoder(self.projection)
        
        # Target layer for intervention
        self.target_layer = 6
        
    def load_robust_operator(self):
        """Load the robust sentiment operator"""
        operator_path = Path("models/sentiment_operator/robust_sentiment_operator.pt")
        
        if not operator_path.exists():
            raise FileNotFoundError("No robust operator found! Run improved Phase 2 first.")
        
        print("Loading robust sentiment operator...")
        
        # Recreate the improved architecture
        class ImprovedSentimentOperator(nn.Module):
            def __init__(self, hd_dim: int = 10000):
                super().__init__()
                self.transform = nn.Sequential(
                    nn.Linear(hd_dim, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, 512),
                    nn.ReLU(), 
                    nn.Dropout(0.1),
                    nn.Linear(512, hd_dim),
                    nn.Tanh()
                )
            
            def forward(self, cogit: torch.Tensor) -> torch.Tensor:
                return self.transform(cogit)
        
        model = ImprovedSentimentOperator()
        checkpoint = torch.load(operator_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✓ Robust operator loaded")
        return model
    
    def load_projection_matrix(self) -> torch.Tensor:
        """Load the same projection matrix used in training"""
        torch.manual_seed(42)  # Same seed as training
        projection = torch.randn(768, 10000)
        projection = projection / torch.norm(projection, dim=0, keepdim=True)
        print("✓ Projection matrix loaded")
        return projection
    
    def encode_to_cogit(self, activation: torch.Tensor) -> torch.Tensor:
        """Encode activation to HDC cogit (same as training)"""
        # Handle shape: (batch, seq, dim) -> (dim,)
        if len(activation.shape) == 3:
            activation = activation.mean(dim=[0, 1])
        elif len(activation.shape) == 2:
            activation = activation.mean(dim=0)
        
        # Project to HD space
        hd_vector = torch.matmul(activation, self.projection)
        cogit = torch.sign(hd_vector)
        return cogit
    
    def decode_from_cogit(self, cogit: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """Decode cogit back to activation space with blending"""
        # Decode to activation vector
        activation = self.decoder.decode_cogit(cogit)
        
        # Reshape to match original
        if len(original_shape) == 3:
            batch_size, seq_len, dim = original_shape
            activation = activation.unsqueeze(0).unsqueeze(0)
            activation = activation.expand(batch_size, seq_len, dim)
        
        return activation
    
    def run_baseline(self, prompt: str, max_tokens: int = 25) -> str:
        """Run GPT-2 without intervention"""
        print(f"\n📝 Baseline:")
        print(f"   '{prompt}'", end="")
        
        tokens = self.adapter.model.to_tokens(prompt)
        output = self.adapter.model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            stop_at_eos=True,
            verbose=False
        )
        
        generated = self.adapter.model.tokenizer.decode(output[0])
        continuation = generated[len(prompt):].strip()
        
        print(f" → {continuation}")
        return continuation
    
    def run_with_gentle_intervention(self, prompt: str, max_tokens: int = 25, 
                                    blend_ratio: float = 0.1) -> str:
        """Run with gentle intervention - blend modified and original activations"""
        print(f"\n🔧 With intervention (blend={blend_ratio}):")
        print(f"   '{prompt}'", end="")
        
        def gentle_intervention(activations, hook):
            """Apply gentle sentiment shift by blending activations"""
            
            original_shape = activations.shape
            
            # Encode current activation to cogit
            cogit = self.encode_to_cogit(activations.cpu())
            
            # Apply the robust operator
            modified_cogit = self.operator(cogit.unsqueeze(0)).squeeze()
            modified_cogit = torch.sign(modified_cogit)
            
            # Decode back to activation space
            modified_activation = self.decode_from_cogit(modified_cogit, original_shape)
            
            # GENTLE BLENDING: Mix original and modified activations
            blended = (1 - blend_ratio) * activations + blend_ratio * modified_activation.to(activations.device)
            
            return blended
        
        # Run with gentle intervention
        hook_name = f"blocks.{self.target_layer}.hook_resid_post"
        
        tokens = self.adapter.model.to_tokens(prompt)
        
        with self.adapter.model.hooks(fwd_hooks=[(hook_name, gentle_intervention)]):
            output = self.adapter.model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                stop_at_eos=True,
                verbose=False
            )
        
        generated = self.adapter.model.tokenizer.decode(output[0])
        continuation = generated[len(prompt):].strip()
        
        print(f" → {continuation}")
        return continuation


class HDCDecoder:
    """Decoder for HDC cogits (same as before)"""
    
    def __init__(self, encoder_projection: torch.Tensor):
        self.projection = encoder_projection
        self.inverse_projection = torch.pinverse(encoder_projection)
        
    def decode_cogit(self, cogit: torch.Tensor) -> torch.Tensor:
        if cogit.dim() == 1:
            cogit = cogit.unsqueeze(0)
        activation = torch.matmul(cogit, self.inverse_projection)
        return activation.squeeze()


def run_improved_experiment():
    """Run the improved sentiment intervention experiment"""
    
    print("\n" + "="*70)
    print("IMPROVED EXPERIMENT: Testing Robust Sentiment Control")
    print("="*70)
    
    # Initialize improved system
    system = ImprovedInterventionSystem()
    
    # Test on diverse neutral prompts
    test_prompts = [
        "The meeting this afternoon will",
        "I opened the envelope and found",
        "The restaurant downtown is",
        "My friend called to say",
        "The project manager announced that",
        "Looking at the data, it appears",
        "The weather report indicates"
    ]
    
    results = []
    
    # Test different intervention strengths
    blend_ratios = [0.05, 0.1, 0.2]  # Start gentle
    
    for prompt in test_prompts[:3]:  # Test first 3 prompts
        print("\n" + "-"*60)
        print(f"TESTING: '{prompt}'")
        print("-"*60)
        
        # Baseline
        baseline_output = system.run_baseline(prompt)
        
        # Test different intervention strengths
        interventions = {}
        for ratio in blend_ratios:
            intervened_output = system.run_with_gentle_intervention(prompt, blend_ratio=ratio)
            interventions[ratio] = intervened_output
        
        # Store results
        results.append({
            'prompt': prompt,
            'baseline': baseline_output,
            'interventions': interventions
        })
        
        # Quick sentiment analysis
        print(f"\n📊 Analysis:")
        negative_indicators = ['bad', 'wrong', 'terrible', 'horrible', 'problem', 
                              'difficult', 'unfortunately', 'disappointing', 'sad', 'angry']
        
        baseline_negative = sum(1 for word in negative_indicators if word in baseline_output.lower())
        
        for ratio, output in interventions.items():
            intervened_negative = sum(1 for word in negative_indicators if word in output.lower())
            change = "↗️ More negative" if intervened_negative > baseline_negative else "→ Similar" if intervened_negative == baseline_negative else "↘️ Less negative"
            coherent = "✅ Coherent" if len(output.split()) > 3 and not any(char in output for char in "——————") else "❌ Broken"
            print(f"   Blend {ratio}: {change}, {coherent}")
    
    print(f"\n" + "="*70)
    print("IMPROVED EXPERIMENT RESULTS")
    print("="*70)
    
    print("\n🎯 KEY QUESTION: Does the improved operator...")
    print("   1. Preserve text coherence? (No more dashes/gibberish)")
    print("   2. Shift sentiment subtly?")
    print("   3. Allow controlled manipulation?")
    
    # Save results
    output_dir = Path("results/sentiment_intervention")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "improved_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to improved_results.json")
    
    return results


if __name__ == "__main__":
    results = run_improved_experiment()