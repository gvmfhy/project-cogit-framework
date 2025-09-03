#!/usr/bin/env python3
"""
Model adapter pattern for abstracting different transformer models.
Provides a unified interface for GPT-2, Llama, and other transformers.
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel
import numpy as np
import random

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


class GPT2Adapter(ModelAdapter):
    """Adapter for GPT-2 family models"""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.lm_model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)  # For generation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store model config
        self.config = self.model.config
        self.hidden_dim = self.config.hidden_size
        self.num_layers = self.config.n_layer
        
        # Hook storage
        self.hooks = []
        self.activations = {}
        
    def extract_hidden_states(self, text: str, layers: List[int]) -> Dict[int, torch.Tensor]:
        """Extract hidden states from specified layers during forward pass"""
        self.activations = {}
        
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                # GPT-2 outputs a tuple, we want the hidden states
                hidden_states = output[0] if isinstance(output, tuple) else output
                # CRITICAL FIX: Use detach().clone() to avoid MPS deadlock
                self.activations[layer_idx] = hidden_states.detach().clone().cpu()
            return hook_fn
        
        # Register hooks
        for layer_idx in layers:
            if layer_idx < self.num_layers:
                # Access GPT-2 transformer blocks
                layer = self.model.h[layer_idx]
                hook = layer.register_forward_hook(create_hook(layer_idx))
                self.hooks.append(hook)
        
        # Run forward pass
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        return self.activations
    
    def inject_hidden_states(self, text: str, modified_states: Dict[int, torch.Tensor], layer: int) -> str:
        """Generate text with modified hidden states injected at specified layer"""
        
        # Custom generation loop that works with hooks
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        generated = inputs.input_ids.clone()
        
        max_tokens = 50
        for _ in range(max_tokens):
            # Create injection hook for this step
            def inject_hook(module, input, output):
                if layer in modified_states:
                    hidden = output[0] if isinstance(output, tuple) else output
                    modified = modified_states[layer].detach().clone().to(hidden.device)
                    if modified.shape == hidden.shape:
                        if isinstance(output, tuple):
                            return (modified,) + output[1:]
                        return modified
                return output
            
            # Register hook
            handle = self.lm_model.transformer.h[layer].register_forward_hook(inject_hook)
            
            # Get next token
            with torch.no_grad():
                outputs = self.lm_model(generated)
                logits = outputs.logits
                next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
            # Remove hook immediately
            handle.remove()
            
            # Add token
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names"""
        return [f"layer_{i}" for i in range(self.num_layers)]
    
    def get_hidden_dim(self) -> int:
        """Get hidden dimension size"""
        return self.hidden_dim


class LlamaAdapter(ModelAdapter):
    """Adapter for Llama family models"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device)
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Store model config
        self.config = self.model.config
        self.hidden_dim = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        
        # Hook storage
        self.hooks = []
        self.activations = {}
        
    def extract_hidden_states(self, text: str, layers: List[int]) -> Dict[int, torch.Tensor]:
        """Extract hidden states from specified layers"""
        self.activations = {}
        
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                # Llama outputs hidden states directly
                hidden_states = output[0] if isinstance(output, tuple) else output
                # CRITICAL FIX: Use detach().clone() to avoid MPS deadlock
                self.activations[layer_idx] = hidden_states.detach().clone().cpu()
            return hook_fn
        
        # Register hooks for Llama layers
        for layer_idx in layers:
            if layer_idx < self.num_layers:
                layer = self.model.model.layers[layer_idx]
                hook = layer.register_forward_hook(create_hook(layer_idx))
                self.hooks.append(hook)
        
        # Run forward pass
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        return self.activations
    
    def inject_hidden_states(self, text: str, modified_states: Dict[int, torch.Tensor], layer: int) -> str:
        """Generate text with modified hidden states"""
        # Similar implementation to GPT-2 but adapted for Llama architecture
        # Implementation details would be similar but with Llama-specific layer access
        raise NotImplementedError("Llama injection to be implemented when running on GPU")
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names"""
        return [f"layer_{i}" for i in range(self.num_layers)]
    
    def get_hidden_dim(self) -> int:
        """Get hidden dimension size"""
        return self.hidden_dim


class ModelAdapterFactory:
    """Factory for creating model adapters based on model name"""
    
    @staticmethod
    def create_adapter(model_name: str, device: str = None) -> ModelAdapter:
        """Create appropriate adapter for the given model"""
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine adapter based on model name
        if "gpt2" in model_name.lower():
            return GPT2Adapter(model_name, device)
        elif "llama" in model_name.lower():
            return LlamaAdapter(model_name, device)
        else:
            # Default to GPT-2 adapter for unknown models
            # Could extend this to support more model families
            return GPT2Adapter(model_name, device)


def test_adapter():
    """Test the model adapter with GPT-2"""
    print("Testing Model Adapter with GPT-2...")
    
    # Create adapter
    adapter = ModelAdapterFactory.create_adapter("gpt2", "cpu")
    print(f"Model loaded: hidden_dim={adapter.get_hidden_dim()}")
    
    # Test extraction
    text = "I think the weather"
    layers = [5, 6, 7]
    hidden_states = adapter.extract_hidden_states(text, layers)
    
    for layer, states in hidden_states.items():
        print(f"Layer {layer}: shape={states.shape}")
    
    # Test injection (simplified)
    modified_states = {6: hidden_states[6] * 1.5}  # Amplify layer 6
    output = adapter.inject_hidden_states(text, modified_states, 6)
    print(f"Original: {text}")
    print(f"Generated: {output}")
    
    print("✓ Adapter test complete")


if __name__ == "__main__":
    test_adapter()