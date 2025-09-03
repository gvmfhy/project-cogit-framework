#!/usr/bin/env python3
"""Absolutely minimal test of the issue"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Testing minimal generation...")

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Just try to generate
text = "Hello"
inputs = tokenizer(text, return_tensors='pt')

print("Generating...")
with torch.no_grad():
    # Use the most basic generation
    output = model.generate(inputs.input_ids, max_length=10)
    
print("Generated:", tokenizer.decode(output[0]))
print("Done!")