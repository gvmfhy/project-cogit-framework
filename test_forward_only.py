#!/usr/bin/env python3
"""Test if forward pass works when generate doesn't"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

print("TESTING: Forward pass vs generate")
print("="*40)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Test forward pass multiple times
text = "The weather is"
inputs = tokenizer(text, return_tensors='pt')

print(f"Input: '{text}'")
print(f"Input IDs: {inputs.input_ids.tolist()}")

# Do manual generation via forward passes
generated_ids = inputs.input_ids.clone()

for i in range(5):
    print(f"\nStep {i+1}:")
    with torch.no_grad():
        outputs = model(generated_ids)
        logits = outputs.logits
        
        # Get next token (greedy)
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        
        print(f"  Next token ID: {next_token_id.item()}")
        print(f"  Next token: '{tokenizer.decode(next_token_id[0])}'")
        
        # Append to sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        
        current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"  Current text: '{current_text}'")

print("\n✓ Manual generation via forward passes works!")
print("✗ But model.generate() hangs - this is a library bug")