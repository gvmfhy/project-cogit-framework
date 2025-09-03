#!/usr/bin/env python3
"""Fix the injection mechanism properly"""

import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
import numpy as np

class WorkingInjectionSystem:
    """A properly working injection system for GPT-2"""
    
    def __init__(self, model_name='gpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.base_model = GPT2Model.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def extract_activation(self, text, layer_idx):
        """Extract activation from a specific layer"""
        inputs = self.tokenizer(text, return_tensors='pt')
        
        activations = {}
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[layer_idx] = hidden.detach()
        
        # Register hook on base model (not LM head)
        handle = self.base_model.h[layer_idx].register_forward_hook(hook)
        
        with torch.no_grad():
            _ = self.base_model(**inputs)
        
        handle.remove()
        return activations[layer_idx]
    
    def generate_with_injection(self, text, modified_activation, layer_idx, max_tokens=20):
        """Generate text with modified activation injected"""
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids']
        
        # Manual generation loop to avoid hook issues
        generated = input_ids.clone()
        
        for _ in range(max_tokens):
            # Create a fresh hook for each token
            def inject_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    # Only inject if shapes match exactly
                    if hidden.shape == modified_activation.shape:
                        return (modified_activation,) + output[1:]
                return output
            
            # Register hook
            handle = self.model.transformer.h[layer_idx].register_forward_hook(inject_hook)
            
            with torch.no_grad():
                # Get logits for next token
                outputs = self.model(generated)
                logits = outputs.logits
                
                # Get next token (greedy for reproducibility)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
            # Remove hook after each step
            handle.remove()
            
            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

def test_working_injection():
    """Test that injection actually changes behavior"""
    print("🔧 Testing Fixed Injection System")
    print("="*40)
    
    system = WorkingInjectionSystem()
    
    # Test texts
    test_prompts = [
        "I think the weather",
        "The president said",
        "Science shows that"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Extract original activation
        layer = 6
        original_act = system.extract_activation(prompt, layer)
        print(f"  Activation shape: {original_act.shape}")
        
        # Generate normally
        inputs = system.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            normal_output = system.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                pad_token_id=system.tokenizer.eos_token_id
            )
        normal_text = system.tokenizer.decode(normal_output[0], skip_special_tokens=True)
        print(f"  Normal: '{normal_text}'")
        
        # Generate with amplified activation (more certain)
        amplified_act = original_act * 1.5
        amplified_text = system.generate_with_injection(prompt, amplified_act, layer, max_tokens=15)
        print(f"  Amplified: '{amplified_text}'")
        
        # Generate with dampened activation (less certain)
        dampened_act = original_act * 0.5
        dampened_text = system.generate_with_injection(prompt, dampened_act, layer, max_tokens=15)
        print(f"  Dampened: '{dampened_text}'")
        
        # Check if we got different outputs
        if normal_text != amplified_text or normal_text != dampened_text:
            print("  ✅ Behavioral change detected!")
        else:
            print("  ⚠️ No behavioral change")

if __name__ == "__main__":
    test_working_injection()