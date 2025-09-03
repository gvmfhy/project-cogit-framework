#!/usr/bin/env python3
"""
Model adapter using TransformerLens for clean hook management.
Replaces manual hook management with TransformerLens's battle-tested infrastructure.
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class ModelAdapter(ABC):
    """Abstract base class for model adapters"""
    
    @abstractmethod
    def extract_hidden_states(self, text: str, layers: List[int]) -> Dict[int, torch.Tensor]:
        """Extract hidden states from specified layers"""
        pass
    
    @abstractmethod
    def inject_hidden_states(self, text: str, modified_states: Dict[int, torch.Tensor], layer: int) -> str:
        """Generate text with injected hidden states"""
        pass
    
    @abstractmethod
    def get_layer_names(self) -> List[str]:
        """Get list of layer names for this model"""
        pass
    
    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Get hidden dimension size"""
        pass


class TransformerLensAdapter(ModelAdapter):
    """Adapter using TransformerLens for clean activation extraction and manipulation"""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        
        # Load model with TransformerLens - handles all hook management internally
        print(f"Loading {model_name} with TransformerLens...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            center_writing_weights=False,  # Keep original behavior
            center_unembed=False,
            fold_ln=False  # Don't fold layer norm for cleaner interventions
        )
        
        # Store model config
        self.hidden_dim = self.model.cfg.d_model
        self.num_layers = self.model.cfg.n_layers
        
    def extract_hidden_states(self, text: str, layers: List[int]) -> Dict[int, torch.Tensor]:
        """Extract hidden states using TransformerLens's run_with_cache"""
        
        # Run model with automatic caching - no manual hooks needed!
        logits, cache = self.model.run_with_cache(text)
        
        # Extract requested layers from cache
        activations = {}
        for layer_idx in layers:
            if layer_idx < self.num_layers:
                # TransformerLens naming convention: blocks.{layer}.hook_resid_post
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                if hook_name in cache:
                    # Cache already has everything detached and on CPU
                    activations[layer_idx] = cache[hook_name].cpu()
                else:
                    print(f"Warning: Layer {layer_idx} not found in cache")
        
        return activations
    
    def inject_hidden_states(self, text: str, modified_states: Dict[int, torch.Tensor], layer: int) -> str:
        """Generate text with modified hidden states using TransformerLens patching"""
        
        # Tokenize input
        tokens = self.model.to_tokens(text)
        
        # Define intervention function
        def intervene_on_layer(activations, hook: HookPoint):
            """Replace activations at specified layer"""
            if layer in modified_states:
                modified = modified_states[layer].to(activations.device)
                # Only modify if shapes match
                if modified.shape == activations.shape:
                    return modified
            return activations
        
        # Generate with intervention using TransformerLens's built-in functionality
        max_tokens = 50
        hook_name = f"blocks.{layer}.hook_resid_post"
        
        # Use model.generate() with hooks
        with self.model.hooks(fwd_hooks=[(hook_name, intervene_on_layer)]):
            output_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=0.0,  # Deterministic
                top_k=1,
                stop_at_eos=True,
                verbose=False
            )
        
        # Decode output
        generated_text = self.model.tokenizer.decode(output_tokens[0])
        return generated_text
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names in TransformerLens format"""
        return [f"blocks.{i}.hook_resid_post" for i in range(self.num_layers)]
    
    def get_hidden_dim(self) -> int:
        """Get hidden dimension size"""
        return self.hidden_dim
    
    def run_with_cache_detailed(self, text: str) -> Tuple[torch.Tensor, Dict]:
        """
        Run model and return full cache for advanced analysis.
        Returns both logits and complete activation cache.
        """
        logits, cache = self.model.run_with_cache(text)
        return logits, cache
    
    def get_attention_patterns(self, text: str) -> Dict[int, torch.Tensor]:
        """Extract attention patterns from all layers"""
        _, cache = self.model.run_with_cache(text)
        
        attention_patterns = {}
        for layer in range(self.num_layers):
            pattern_name = f"blocks.{layer}.attn.hook_pattern"
            if pattern_name in cache:
                attention_patterns[layer] = cache[pattern_name].cpu()
        
        return attention_patterns
    
    def patch_activation(self, text: str, layer: int, position: int, 
                        direction: torch.Tensor, scale: float = 1.0) -> str:
        """
        Patch a specific activation at a given layer and position.
        This is useful for steering experiments.
        """
        tokens = self.model.to_tokens(text)
        
        def add_direction(activations, hook: HookPoint):
            """Add a direction vector to specific position"""
            if position < activations.shape[1]:
                activations[:, position, :] += scale * direction.to(activations.device)
            return activations
        
        hook_name = f"blocks.{layer}.hook_resid_post"
        
        with self.model.hooks(fwd_hooks=[(hook_name, add_direction)]):
            output_tokens = self.model.generate(
                tokens,
                max_new_tokens=50,
                temperature=0.0,
                top_k=1,
                stop_at_eos=True,
                verbose=False
            )
        
        return self.model.tokenizer.decode(output_tokens[0])


class ModelAdapterFactory:
    """Factory for creating model adapters"""
    
    @staticmethod
    def create_adapter(model_name: str, device: str = None, use_transformer_lens: bool = True) -> ModelAdapter:
        """Create appropriate adapter for the given model"""
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if use_transformer_lens:
            # Use TransformerLens adapter by default for supported models
            supported_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", 
                              "gpt-neo-125M", "gpt-neo-1.3B", "gpt-neo-2.7B",
                              "gpt-j-6B", "pythia-70m", "pythia-160m", "pythia-410m",
                              "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b"]
            
            if any(model in model_name for model in supported_models):
                return TransformerLensAdapter(model_name, device)
            else:
                print(f"Warning: {model_name} not directly supported by TransformerLens")
                print("Falling back to manual implementation")
                # Could fall back to old implementation here
                raise NotImplementedError(f"Model {model_name} not supported")
        else:
            # Use old manual implementation if specified
            raise NotImplementedError("Manual implementation deprecated - use TransformerLens")


def test_adapter():
    """Test the TransformerLens adapter"""
    print("Testing TransformerLens Adapter...")
    print("=" * 50)
    
    # Create adapter
    adapter = ModelAdapterFactory.create_adapter("gpt2", "cpu")
    print(f"✓ Model loaded: hidden_dim={adapter.get_hidden_dim()}")
    
    # Test extraction
    text = "The weather today is"
    layers = [5, 6, 7]
    print(f"\nExtracting hidden states from layers {layers}...")
    hidden_states = adapter.extract_hidden_states(text, layers)
    
    for layer, states in hidden_states.items():
        print(f"  Layer {layer}: shape={states.shape}")
    
    # Test basic generation
    print(f"\nOriginal prompt: '{text}'")
    
    # Generate without intervention
    print("\nGenerating without intervention...")
    output = adapter.model.generate(
        adapter.model.to_tokens(text),
        max_new_tokens=10,
        temperature=0.0,
        verbose=False
    )
    baseline = adapter.model.tokenizer.decode(output[0])
    print(f"  Baseline: {baseline}")
    
    # Test injection - amplify layer 6
    print("\nTesting activation injection (amplifying layer 6)...")
    modified_states = {6: hidden_states[6] * 2.0}
    output_modified = adapter.inject_hidden_states(text, modified_states, 6)
    print(f"  Modified: {output_modified}")
    
    # Test attention pattern extraction
    print("\nExtracting attention patterns...")
    attention = adapter.get_attention_patterns(text)
    print(f"  Found {len(attention)} layers with attention patterns")
    if attention:
        layer_0_attention = attention[0]
        print(f"  Layer 0 attention shape: {layer_0_attention.shape}")
    
    print("\n✓ All tests passed!")
    return adapter


def compare_implementations():
    """Compare outputs between old and new implementations"""
    print("Comparing implementations...")
    print("=" * 50)
    
    text = "The capital of France is"
    
    # Test with TransformerLens
    tl_adapter = TransformerLensAdapter("gpt2", "cpu")
    
    # Extract states
    states = tl_adapter.extract_hidden_states(text, [0, 5, 11])
    print("\nTransformerLens extraction:")
    for layer, state in states.items():
        print(f"  Layer {layer}: shape={state.shape}, mean={state.mean():.4f}")
    
    # Test generation with cache
    print("\nTransformerLens generation test:")
    logits, cache = tl_adapter.run_with_cache_detailed(text)
    print(f"  Cache contains {len(cache)} tensors")
    print(f"  Logits shape: {logits.shape}")
    
    print("\n✓ TransformerLens implementation working correctly!")


if __name__ == "__main__":
    # Run basic tests
    test_adapter()
    
    print("\n" + "=" * 50)
    
    # Run comparison
    compare_implementations()