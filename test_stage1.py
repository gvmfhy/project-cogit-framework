#!/usr/bin/env python3
"""Test Stage 1 functionality"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import sys
sys.path.append('.')

from src.model_adapter import GPT2Adapter
import torch

print("Testing Stage 1 components...")

# Create adapter
print("Loading GPT-2...")
adapter = GPT2Adapter("gpt2", "cpu")
print(f"✓ Model loaded: {adapter.get_hidden_dim()}D")

# Test extraction
text = "I think the weather"
layers = [5, 6, 7]
print(f"\nExtracting from: '{text}'")
states = adapter.extract_hidden_states(text, layers)

for layer, state in states.items():
    print(f"  Layer {layer}: shape={state.shape}")

print("\n✓ Extraction works!")

# Test generation
print("\nTesting generation...")
inputs = adapter.tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = adapter.lm_model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0.7,
        do_sample=True
    )
generated = adapter.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: '{generated}'")

print("\n✓ All tests pass!")