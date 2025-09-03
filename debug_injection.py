#!/usr/bin/env python3
"""Root cause analysis - find exactly where the hang occurs"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("ROOT CAUSE ANALYSIS")
print("="*40)

# 1. Load model and tokenizer on CPU
device = torch.device("cpu")
print("1. Loading GPT-2 on CPU...")
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
print("   ✓ Model loaded")

# 2. Define simple hook that replaces with zeros
def zero_hook(module, input, output):
    """Replace output with zeros of same shape"""
    if isinstance(output, tuple):
        hidden = output[0]
        zeros = torch.zeros_like(hidden)
        print(f"   Hook: Replacing tensor shape {hidden.shape} with zeros")
        return (zeros,) + output[1:]
    else:
        zeros = torch.zeros_like(output)
        print(f"   Hook: Replacing tensor shape {output.shape} with zeros")
        return zeros

# 3. Test input
text = "Hello world"
inputs = tokenizer(text, return_tensors='pt').to(device)
print(f"\n2. Input: '{text}'")
print(f"   Input IDs shape: {inputs.input_ids.shape}")

# 4. FIRST TEST: Forward pass with hook
print("\n3. TEST 1: Forward pass with hook on layer 6...")
hook_handle = model.transformer.h[6].register_forward_hook(zero_hook)

try:
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
        print(f"   ✓ Forward pass succeeded")
        print(f"   Output logits shape: {logits.shape}")
        # Check if zeros had effect (logits should be unusual)
        print(f"   Max logit: {logits.max().item():.3f}")
        print(f"   Min logit: {logits.min().item():.3f}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
finally:
    hook_handle.remove()

# 5. SECOND TEST: Generate with hook
print("\n4. TEST 2: Generate with hook on layer 6...")
hook_handle = model.transformer.h[6].register_forward_hook(zero_hook)

try:
    with torch.no_grad():
        print("   Calling generate()...")
        output = model.generate(
            **inputs,  # Use full tokenizer output with attention_mask
            max_new_tokens=5,
            do_sample=False,  # Deterministic
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"   ✓ Generate succeeded")
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"   Generated: '{generated_text}'")
except Exception as e:
    print(f"   ✗ Generate failed: {e}")
finally:
    hook_handle.remove()

print("\n" + "="*40)
print("ROOT CAUSE ANALYSIS COMPLETE")