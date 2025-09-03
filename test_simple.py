#!/usr/bin/env python3
"""Simplest possible test - does basic generation work at all?"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("SIMPLEST TEST: Basic generation without hooks")
print("="*40)

# Load model
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print("✓ Loaded")

# Test 1: Forward pass
print("\n1. Forward pass...")
inputs = tokenizer("Hello", return_tensors='pt')
with torch.no_grad():
    output = model(**inputs)
    print(f"✓ Forward pass works. Logits shape: {output.logits.shape}")

# Test 2: Generate with explicit parameters
print("\n2. Generate with max_length=10...")
with torch.no_grad():
    output = model.generate(
        inputs.input_ids,
        max_length=10,  # Use max_length instead of max_new_tokens
        do_sample=False
    )
    print(f"✓ Generate works. Output shape: {output.shape}")
    print(f"Generated: '{tokenizer.decode(output[0])}'")

print("\nBASIC TESTS COMPLETE")