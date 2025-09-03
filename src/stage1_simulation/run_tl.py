#!/usr/bin/env python3
"""
Stage 1: Activation Extraction with TransformerLens
Clean implementation using TransformerLens for stable activation extraction.
No more hook deadlocks!
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import random
import numpy as np
import torch
import json
import jsonlines
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import yaml

# Import our new TransformerLens adapter
from src.model_adapter_tl import TransformerLensAdapter

# Deterministic seeding
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_file = Path("config.yaml")
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    else:
        # Fallback config if file doesn't exist
        return {
            'model': {'name': 'gpt2'},
            'experiment': {
                'extraction_layers': [5, 6, 7],
                'dimensions': [
                    {
                        'name': 'certainty',
                        'low_examples': ['I might think', 'Perhaps it is', 'It could be'],
                        'high_examples': ['I definitely know', 'It is certain that', 'Absolutely']
                    },
                    {
                        'name': 'agreement',
                        'low_examples': ['I disagree that', 'That is wrong', 'No way'],
                        'high_examples': ['I agree that', 'That is correct', 'Exactly right']
                    }
                ]
            },
            'paths': {'mode': 'local'}
        }


def extract_activation_pairs(adapter: TransformerLensAdapter, dimension_config: Dict) -> List[Dict[str, Any]]:
    """
    Extract hidden states from contrasting text pairs using TransformerLens.
    Clean, stable, no hook management needed!
    """
    pairs = []
    
    low_examples = dimension_config['low_examples']
    high_examples = dimension_config['high_examples']
    dimension_name = dimension_config['name']
    
    # Get extraction layers from config
    config = load_config()
    layers = config['experiment']['extraction_layers']
    
    print(f"Extracting {dimension_name} dimension pairs...")
    
    for low_text in low_examples:
        # Extract with TransformerLens - clean and simple!
        low_states = adapter.extract_hidden_states(low_text, layers)
        
        for high_text in high_examples:
            high_states = adapter.extract_hidden_states(high_text, layers)
            
            # Create pair for each layer
            for layer_idx in layers:
                if layer_idx in low_states and layer_idx in high_states:
                    pair = {
                        'dimension': dimension_name,
                        'layer': layer_idx,
                        'low_text': low_text,
                        'high_text': high_text,
                        'low_activation': low_states[layer_idx].cpu().numpy().tolist(),
                        'high_activation': high_states[layer_idx].cpu().numpy().tolist(),
                        'timestamp': datetime.now().isoformat()
                    }
                    pairs.append(pair)
    
    return pairs


def run_extraction_pipeline():
    """Run the full extraction pipeline with TransformerLens"""
    print("=" * 60)
    print("STAGE 1: ACTIVATION EXTRACTION (TransformerLens)")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Initialize TransformerLens adapter
    print("\nInitializing TransformerLens adapter...")
    model_name = config['model']['name']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = TransformerLensAdapter(model_name, device)
    print(f"✓ Model loaded: {model_name} (hidden_dim={adapter.get_hidden_dim()})")
    
    # Extract pairs for each dimension
    all_pairs = []
    for dimension_config in config['experiment']['dimensions']:
        pairs = extract_activation_pairs(adapter, dimension_config)
        all_pairs.extend(pairs)
        print(f"  ✓ Extracted {len(pairs)} pairs for {dimension_config['name']}")
    
    # Save results
    output_dir = Path("data/raw/sims")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"activations_{timestamp}.jsonl"
    
    with jsonlines.open(output_file, 'w') as writer:
        for pair in all_pairs:
            writer.write(pair)
    
    print(f"\n✓ Saved {len(all_pairs)} activation pairs to {output_file}")
    
    # Also save metadata
    metadata = {
        'model': model_name,
        'layers': config['experiment']['extraction_layers'],
        'dimensions': [d['name'] for d in config['experiment']['dimensions']],
        'total_pairs': len(all_pairs),
        'timestamp': timestamp
    }
    
    metadata_file = output_dir / f"metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_file}")
    
    return output_file, all_pairs


def demonstrate_stability():
    """Demonstrate that TransformerLens doesn't hang on generation with hooks"""
    print("\n" + "=" * 60)
    print("STABILITY TEST: Generation with Activation Injection")
    print("=" * 60)
    
    adapter = TransformerLensAdapter("gpt2", "cpu")
    
    # Extract states
    text = "The concept of artificial intelligence"
    states = adapter.extract_hidden_states(text, [5, 6, 7])
    print(f"\n1. Extracted states from layers [5, 6, 7]")
    
    # Inject and generate - THIS WOULD HANG WITH OLD IMPLEMENTATION
    modified_states = {6: states[6] * 1.5}
    
    print(f"\n2. Injecting amplified activations at layer 6...")
    print("   (This would hang indefinitely with the old implementation)")
    
    output = adapter.inject_hidden_states(text, modified_states, 6)
    print(f"\n3. ✓ SUCCESS! Generated text with injection:")
    print(f"   {output[:100]}...")
    
    print("\n✓ No hangs, no deadlocks - TransformerLens handles everything!")


if __name__ == "__main__":
    # Run the extraction pipeline
    output_file, pairs = run_extraction_pipeline()
    
    # Demonstrate stability
    demonstrate_stability()
    
    print("\n" + "=" * 60)
    print("Stage 1 Complete - Ready for Stage 2 (Encoding)")
    print("=" * 60)