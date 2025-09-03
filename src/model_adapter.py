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
from transformers import AutoModel, AutoTokenizer
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
                self.activations[layer_idx] = hidden_states.detach().cpu()
            return hook_fn
        
        # Register hooks
        for layer_idx in layers:
            if layer_idx < self.num_layers:
                # Access GPT-2 transformer blocks
                layer = self.model.transformer.h[layer_idx]
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
        
        # This is complex - we need to modify forward pass
        # For now, implementing a simplified version
        class ModifiedGPT2(nn.Module):
            def __init__(self, original_model, modified_states, target_layer):
                super().__init__()
                self.model = original_model
                self.modified_states = modified_states
                self.target_layer = target_layer
                self.hook_handle = None
                
            def modify_hidden_states(self, module, input, output):
                # Replace hidden states at target layer
                if self.target_layer in self.modified_states:
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    # Inject our modified states
                    modified = self.modified_states[self.target_layer].to(hidden_states.device)
                    # Ensure shapes match
                    if modified.shape == hidden_states.shape:
                        if isinstance(output, tuple):
                            return (modified,) + output[1:]
                        return modified
                return output
                
            def forward(self, **kwargs):
                # Register modification hook
                if self.target_layer < len(self.model.transformer.h):
                    self.hook_handle = self.model.transformer.h[self.target_layer].register_forward_hook(
                        self.modify_hidden_states
                    )
                
                # Generate
                outputs = self.model.generate(
                    **kwargs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.model.config.eos_token_id
                )
                
                # Clean up hook
                if self.hook_handle:
                    self.hook_handle.remove()
                    
                return outputs
        
        # Prepare input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Create modified model and generate
        modified_model = ModifiedGPT2(self.model, modified_states, layer)
        with torch.no_grad():
            output_ids = modified_model(**inputs)
        
        # Decode output
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
    
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
                self.activations[layer_idx] = hidden_states.detach().cpu()
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