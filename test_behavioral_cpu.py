#!/usr/bin/env python3
"""Test behavioral changes - FORCE CPU to avoid MPS issues"""

import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# Force CPU backend
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('.')

import torch
import numpy as np
from src.model_adapter import GPT2Adapter

# FORCE CPU
torch.set_default_device('cpu')

print("🧪 Testing Behavioral Changes (CPU Mode)")
print("="*40)

# Load adapter - explicitly use CPU
adapter = GPT2Adapter("gpt2", device="cpu")
print(f"✓ Model loaded on device: cpu")

# Test prompts
test_prompts = [
    "I think the weather",
    "The scientists believe",
    "Research shows that"
]

for prompt in test_prompts:
    print(f"\n📝 Prompt: '{prompt}'")
    
    layers = [6]
    
    # Extract original activation
    original_states = adapter.extract_hidden_states(prompt, layers)
    original_act = original_states[6]
    print(f"   Activation shape: {original_act.shape}")
    
    # Generate with original
    inputs = adapter.tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        orig_output = adapter.lm_model.generate(
            **inputs,
            max_new_tokens=15,
            temperature=0.7,
            do_sample=False,  # Deterministic for testing
            pad_token_id=adapter.tokenizer.eos_token_id
        )
    orig_text = adapter.tokenizer.decode(orig_output[0], skip_special_tokens=True)
    print(f"   Original: '{orig_text}'")
    
    # Create cognitive manipulations
    manipulations = {
        "Amplified (1.5x)": original_act * 1.5,
        "Strongly amplified (2x)": original_act * 2.0,
        "Dampened (0.5x)": original_act * 0.5,
        "Inverted (-1x)": original_act * -1.0
    }
    
    for name, modified_act in manipulations.items():
        modified_states = {6: modified_act}
        try:
            modified_text = adapter.inject_hidden_states(prompt, modified_states, 6)
            print(f"   {name}: '{modified_text}'")
            
            # Check if different
            if modified_text != orig_text:
                print(f"      ✅ Behavioral change detected!")
        except Exception as e:
            print(f"   {name}: ❌ Error - {e}")

print("\n" + "="*40)
print("✓ Behavioral testing complete!")