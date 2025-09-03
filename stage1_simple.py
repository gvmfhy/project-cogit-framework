#!/usr/bin/env python3
"""Simplified Stage 1 for testing"""

import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('.')

from src.model_adapter import GPT2Adapter
import json
import jsonlines
from datetime import datetime
from pathlib import Path

print("🧠 Stage 1: Real Activation Extraction")
print("="*40)

# Create adapter
print("Loading GPT-2...")
adapter = GPT2Adapter("gpt2", "cpu")
print(f"✓ Loaded: {adapter.get_hidden_dim()}D model")

# Simple test data
dimensions = {
    'certainty': {
        'low': ["I might think", "Perhaps it's"],
        'high': ["I definitely know", "It's certain that"]
    }
}

# Extract activations
layers = [5, 6, 7]
pairs = []

print("\nExtracting activation pairs...")
for dim_name, examples in dimensions.items():
    for low_text in examples['low']:
        low_states = adapter.extract_hidden_states(low_text, layers)
        
        for high_text in examples['high']:
            high_states = adapter.extract_hidden_states(high_text, layers)
            
            for layer in layers:
                pairs.append({
                    'dimension': dim_name,
                    'layer': layer,
                    'low_text': low_text,
                    'high_text': high_text,
                    'low_activation': low_states[layer].cpu().numpy().tolist(),
                    'high_activation': high_states[layer].cpu().numpy().tolist(),
                    'timestamp': datetime.now().isoformat()
                })

print(f"✓ Extracted {len(pairs)} pairs")

# Save
output_dir = Path("data/raw/activations")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"activation_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(pairs)

print(f"✓ Saved to {output_file}")

# Save metrics
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
metrics = {
    'model': 'gpt2',
    'num_pairs': len(pairs),
    'layers': layers,
    'timestamp': datetime.now().isoformat()
}
with open(results_dir / "stage1_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✓ Stage 1 complete!")