#!/usr/bin/env python3
"""
Phase 3: Testing the Intervention - Does the Dial Work?

The crucial experiment: Can we use our learned operator to control GPT-2's sentiment?
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
print("PHASE 3: TESTING THE SENTIMENT INTERVENTION")
print("Can we control GPT-2's output sentiment?")
print("=" * 70)


class HDCDecoder:
    """Decodes HDC cogits back to activation space"""
    
    def __init__(self, encoder_projection: torch.Tensor):
        """Initialize with the inverse of the encoder's projection"""
        self.projection = encoder_projection
        # Compute pseudo-inverse for decoding
        self.inverse_projection = torch.pinverse(encoder_projection)
        
    def decode_cogit(self, cogit: torch.Tensor) -> torch.Tensor:
        """Decode a cogit back to activation space"""
        # Ensure cogit is the right shape
        if cogit.dim() == 1:
            cogit = cogit.unsqueeze(0)
        # Project from HD space back to activation space
        # cogit is (1, 10000), inverse_projection is (10000, 768)
        activation = torch.matmul(cogit, self.inverse_projection)
        return activation.squeeze()


class SentimentInterventionSystem:
    """Complete system for sentiment intervention in GPT-2"""
    
    def __init__(self):
        print("\n[Initializing Intervention System]")
        
        # Load GPT-2 with TransformerLens
        print("Loading GPT-2...")
        self.adapter = TransformerLensAdapter("gpt2", "cpu")
        print(f"✓ Model loaded")
        
        # Load the trained operator
        self.operator = self.load_operator()
        
        # Load encoder projection matrix for encoding/decoding
        self.projection = self.load_projection_matrix()
        self.decoder = HDCDecoder(self.projection)
        
        # Target layer for intervention
        self.target_layer = 6
        
    def load_operator(self):
        """Load the trained make_negative_operator"""
        operator_path = Path("models/sentiment_operator/make_negative_operator.pt")
        
        if not operator_path.exists():
            raise FileNotFoundError("No trained operator found! Run Phase 2 first.")
        
        print("Loading sentiment operator...")
        
        # Recreate the model architecture
        class SimpleSentimentOperator(nn.Module):
            def __init__(self, hd_dim: int = 10000):
                super().__init__()
                self.transform = nn.Sequential(
                    nn.Linear(hd_dim, 512),
                    nn.Tanh(),
                    nn.Linear(512, hd_dim),
                    nn.Tanh()
                )
            
            def forward(self, cogit: torch.Tensor) -> torch.Tensor:
                return self.transform(cogit)
        
        model = SimpleSentimentOperator()
        checkpoint = torch.load(operator_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✓ Operator loaded")
        return model
    
    def load_projection_matrix(self) -> torch.Tensor:
        """Load or recreate the HDC projection matrix"""
        # Use same seed as in Phase 1 for consistency
        torch.manual_seed(42)
        projection = torch.randn(768, 10000)
        projection = projection / torch.norm(projection, dim=0, keepdim=True)
        print("✓ Projection matrix loaded")
        return projection
    
    def encode_to_cogit(self, activation: torch.Tensor) -> torch.Tensor:
        """Encode activation to HDC cogit"""
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
        """Decode cogit back to activation space"""
        # Decode to activation vector
        activation = self.decoder.decode_cogit(cogit)
        
        # Reshape to match original
        if len(original_shape) == 3:
            # Broadcast back to (batch, seq, dim)
            batch_size, seq_len, dim = original_shape
            activation = activation.unsqueeze(0).unsqueeze(0)
            activation = activation.expand(batch_size, seq_len, dim)
        
        return activation
    
    def run_baseline(self, prompt: str, max_tokens: int = 30) -> str:
        """Run GPT-2 without intervention (control condition)"""
        print(f"\n📝 Baseline (no intervention):")
        print(f"   Prompt: '{prompt}'")
        
        tokens = self.adapter.model.to_tokens(prompt)
        output = self.adapter.model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=0.7,  # Add some variety
            top_k=50,
            stop_at_eos=True,
            verbose=False
        )
        
        generated = self.adapter.model.tokenizer.decode(output[0])
        continuation = generated[len(prompt):]  # Just the new part
        
        print(f"   Output: '{continuation.strip()}'")
        return continuation
    
    def run_with_intervention(self, prompt: str, max_tokens: int = 30) -> str:
        """Run GPT-2 with sentiment intervention"""
        print(f"\n🔧 With intervention (make negative):")
        print(f"   Prompt: '{prompt}'")
        
        # Define the intervention function
        def make_negative_intervention(activations, hook):
            """Intervention: Apply make_negative_operator to activations"""
            
            # Store original shape
            original_shape = activations.shape
            
            # Encode current activation to cogit
            cogit = self.encode_to_cogit(activations.cpu())
            
            # Apply the make_negative operator
            negative_cogit = self.operator(cogit.unsqueeze(0)).squeeze()
            negative_cogit = torch.sign(negative_cogit)  # Ensure binary
            
            # Decode back to activation space
            modified_activation = self.decode_from_cogit(negative_cogit, original_shape)
            
            # Return modified activation (move to correct device)
            return modified_activation.to(activations.device)
        
        # Run with intervention at target layer
        hook_name = f"blocks.{self.target_layer}.hook_resid_post"
        
        tokens = self.adapter.model.to_tokens(prompt)
        
        with self.adapter.model.hooks(fwd_hooks=[(hook_name, make_negative_intervention)]):
            output = self.adapter.model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_k=50,
                stop_at_eos=True,
                verbose=False
            )
        
        generated = self.adapter.model.tokenizer.decode(output[0])
        continuation = generated[len(prompt):]  # Just the new part
        
        print(f"   Output: '{continuation.strip()}'")
        return continuation


def run_experiment():
    """Run the complete sentiment intervention experiment"""
    
    print("\n" + "="*70)
    print("EXPERIMENT: Testing Sentiment Control")
    print("="*70)
    
    # Initialize the intervention system
    system = SentimentInterventionSystem()
    
    # Test on multiple neutral prompts
    neutral_prompts = [
        "The report is on the table and",
        "I walked into the room and saw",
        "The meeting this morning was",
        "My neighbor told me that",
        "The weather forecast says",
        "The new policy states that",
        "Yesterday I discovered that"
    ]
    
    results = []
    
    for prompt in neutral_prompts:
        print("\n" + "-"*60)
        print(f"TESTING PROMPT: '{prompt}'")
        print("-"*60)
        
        # Run baseline (no intervention)
        baseline_output = system.run_baseline(prompt)
        
        # Run with intervention
        intervened_output = system.run_with_intervention(prompt)
        
        # Store results
        results.append({
            'prompt': prompt,
            'baseline': baseline_output.strip(),
            'intervened': intervened_output.strip()
        })
        
        # Quick sentiment check (you could use a sentiment classifier here)
        print("\n📊 Quick Analysis:")
        negative_words = ['bad', 'wrong', 'terrible', 'horrible', 'awful', 'worst', 
                         'hate', 'angry', 'sad', 'unfortunately', 'problem', 'difficult']
        
        baseline_negative = any(word in baseline_output.lower() for word in negative_words)
        intervened_negative = any(word in intervened_output.lower() for word in negative_words)
        
        if intervened_negative and not baseline_negative:
            print("   ✓ Intervention made output more negative!")
        elif intervened_negative:
            print("   ✓ Intervened output contains negative sentiment")
        else:
            print("   ⚠️  No clear negative shift detected (may need more subtle analysis)")
    
    # Save results
    output_dir = Path("results/sentiment_intervention")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "intervention_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    
    # Summary
    print("\n📋 SUMMARY OF RESULTS:")
    print("-"*40)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Prompt: '{result['prompt']}'")
        print(f"   Baseline:    ...{result['baseline'][:50]}")
        print(f"   Intervened:  ...{result['intervened'][:50]}")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("We have tested whether our learned operator can control GPT-2's sentiment.")
    print("The results show the actual effect of our HDC-based intervention.")
    print("Check 'results/sentiment_intervention/intervention_results.json' for full data.")
    
    return results


if __name__ == "__main__":
    results = run_experiment()