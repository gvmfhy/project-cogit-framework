#!/usr/bin/env python3
"""Simplified Stage 2 - just random projection for now"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('.')

import torch
import numpy as np
import jsonlines
import json
from pathlib import Path
from datetime import datetime

print("🔢 Stage 2: HDC Projection")
print("="*40)

# Load activation pairs
input_dir = Path("data/raw/activations")
pairs_files = list(input_dir.glob("activation_pairs_*.jsonl"))
latest_file = sorted(pairs_files)[-1]

print(f"Loading: {latest_file.name}")
with jsonlines.open(latest_file) as reader:
    pairs = list(reader)

print(f"✓ Loaded {len(pairs)} pairs")

# Simple random projection
input_dim = 768
hdc_dim = 10000

torch.manual_seed(42)
projection_matrix = torch.randn(input_dim, hdc_dim) / np.sqrt(input_dim)

# Project all pairs
encoded_pairs = []
for pair in pairs:
    # Get activations
    low_act = torch.tensor(pair['low_activation'], dtype=torch.float32)
    high_act = torch.tensor(pair['high_activation'], dtype=torch.float32)
    
    # Flatten if needed
    if len(low_act.shape) > 2:
        low_act = low_act.view(-1, input_dim)
    if len(high_act.shape) > 2:
        high_act = high_act.view(-1, input_dim)
    
    # Project to HDC
    low_cogit = torch.tanh(torch.matmul(low_act, projection_matrix))
    high_cogit = torch.tanh(torch.matmul(high_act, projection_matrix))
    
    encoded_pairs.append({
        'dimension': pair['dimension'],
        'layer': pair['layer'],
        'low_text': pair['low_text'],
        'high_text': pair['high_text'],
        'low_cogit': low_cogit.cpu().numpy().tolist(),
        'high_cogit': high_cogit.cpu().numpy().tolist(),
        'projection_strategy': 'random',
        'timestamp': datetime.now().isoformat()
    })

print(f"✓ Encoded {len(encoded_pairs)} pairs")

# Save
output_dir = Path("data/processed/cogits")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"cogits_random_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(encoded_pairs)

print(f"✓ Saved to {output_file}")

# Metrics
results_dir = Path("results")
metrics = {
    'projection': 'random',
    'input_dim': input_dim,
    'hdc_dim': hdc_dim,
    'num_pairs': len(encoded_pairs),
    'timestamp': datetime.now().isoformat()
}
with open(results_dir / "stage2_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✓ Stage 2 complete!")