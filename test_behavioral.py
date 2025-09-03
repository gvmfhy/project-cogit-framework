#!/usr/bin/env python3
"""Test behavioral changes from cognitive manipulation"""

import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('.')

import torch
import numpy as np
from src.model_adapter import GPT2Adapter
from pathlib import Path

print("🧪 Testing Behavioral Changes")
print("="*40)

# Load adapter
adapter = GPT2Adapter("gpt2", "cpu")
print("✓ Model loaded")

# Test prompt
prompt = "I think the weather"
layers = [6]

# Extract original activation
original_states = adapter.extract_hidden_states(prompt, layers)
original_act = original_states[6]
print(f"\nOriginal activation shape: {original_act.shape}")

# Generate with original
inputs = adapter.tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    orig_output = adapter.lm_model.generate(
        **inputs,
        max_new_tokens=15,
        temperature=0.7,
        do_sample=True,
        pad_token_id=adapter.tokenizer.eos_token_id
    )
orig_text = adapter.tokenizer.decode(orig_output[0], skip_special_tokens=True)
print(f"\nOriginal: '{orig_text}'")

# Create a simple manipulation (increase certainty)
# Amplify the activation by 1.5x to simulate "more certain" state
modified_act = original_act * 1.5

# Inject and generate
modified_states = {6: modified_act}
modified_text = adapter.inject_hidden_states(prompt, modified_states, 6)
print(f"Modified: '{modified_text}'")

# Check difference
if orig_text != modified_text:
    print("\n✅ SUCCESS: Behavioral change detected!")
    print("   Injection is working - the model generates different text")
else:
    print("\n⚠️  WARNING: No behavioral change detected")
    print("   Text is identical despite injection")

# Test with stronger manipulation
very_modified_act = original_act * 2.0
very_modified_states = {6: very_modified_act}
very_modified_text = adapter.inject_hidden_states(prompt, very_modified_states, 6)
print(f"\nStronger: '{very_modified_text}'")

print("\n✓ Behavioral testing complete!")