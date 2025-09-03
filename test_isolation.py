#!/usr/bin/env python3
"""Isolate the exact issue with hooks and generation"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

print("ISOLATION TEST")
print("="*40)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("Hello", return_tensors='pt')

# Test 1: Generate without hook
print("\n1. Generate WITHOUT hook...")
try:
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        print(f"✓ Works: {tokenizer.decode(output[0])}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Forward pass WITH hook  
print("\n2. Forward pass WITH hook...")
def dummy_hook(module, input, output):
    return output  # Do nothing

hook = model.transformer.h[6].register_forward_hook(dummy_hook)
try:
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"✓ Works: logits shape {outputs.logits.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
finally:
    hook.remove()

# Test 3: Generate WITH hook (the problem case)
print("\n3. Generate WITH hook...")
print("   (This will likely hang - Ctrl+C to stop)")
hook = model.transformer.h[6].register_forward_hook(dummy_hook)
try:
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        print(f"✓ Works: {tokenizer.decode(output[0])}")
except Exception as e:
    print(f"✗ Failed: {e}")
finally:
    hook.remove()

print("\nCONCLUSION: generate() with hooks is the issue")