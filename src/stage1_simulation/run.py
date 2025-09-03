#!/usr/bin/env python3
"""
Stage 1: Real Activation Extraction
Extracts real hidden states from transformer models for cognitive manipulation experiments.
Model-agnostic design - works with GPT-2, Llama, or any transformer.
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

# Import our model adapter
from src.model_adapter import ModelAdapterFactory

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
                    }
                ]
            },
            'paths': {'mode': 'local'}
        }


def extract_activation_pairs(adapter, dimension_config: Dict) -> List[Dict[str, Any]]:
    """
    Extract hidden states from contrasting text pairs for a cognitive dimension.
    This creates training data for learning manipulation operators.
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
        # Extract hidden states for low-dimension text
        low_states = adapter.extract_hidden_states(low_text, layers)
        
        for high_text in high_examples:
            # Extract hidden states for high-dimension text
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


def generate_probing_texts(base_prompts: List[str], adapter) -> List[Dict[str, Any]]:
    """
    Generate variations of prompts and extract their hidden states.
    This creates a dataset for testing learned operators.
    """
    probing_data = []
    config = load_config()
    layers = config['experiment']['extraction_layers']
    
    for prompt in base_prompts:
        # Extract hidden states
        hidden_states = adapter.extract_hidden_states(prompt, layers)
        
        # Generate continuation with current model state
        inputs = adapter.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            # Use lm_model for generation (GPT2Adapter has this)
            if hasattr(adapter, 'lm_model'):
                outputs = adapter.lm_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=adapter.tokenizer.eos_token_id
                )
            else:
                # Fallback: just use prompt as generated text
                outputs = inputs['input_ids']
        
        generated_text = adapter.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Store probing data
        for layer_idx, states in hidden_states.items():
            probing_data.append({
                'prompt': prompt,
                'generated': generated_text,
                'layer': layer_idx,
                'hidden_states': states.cpu().numpy().tolist(),
                'timestamp': datetime.now().isoformat()
            })
    
    return probing_data


def main():
    """Main extraction pipeline"""
    print("🧠 Stage 1: Real Activation Extraction Pipeline")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    model_name = config['model']['name']
    model_config = config['model']['configs'].get(model_name, {})
    device = model_config.get('device', 'cpu')
    
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Hidden dimension: {model_config.get('hidden_dim', 'unknown')}")
    
    # Create model adapter
    print("\n📚 Loading model...")
    adapter = ModelAdapterFactory.create_adapter(model_name, device)
    print(f"✓ Model loaded: {adapter.get_hidden_dim()}D hidden states")
    
    # Create output directory
    paths_mode = config['paths']['mode']
    base_path = config['paths'][paths_mode]['data_dir']
    output_dir = Path(base_path) / "raw" / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract activation pairs for each cognitive dimension
    all_pairs = []
    all_probing = []
    
    for dimension in config['experiment']['dimensions']:
        # Extract contrasting pairs for learning operators
        pairs = extract_activation_pairs(adapter, dimension)
        all_pairs.extend(pairs)
        print(f"✓ Extracted {len(pairs)} pairs for {dimension['name']}")
        
        # Generate probing data for testing
        all_prompts = dimension['low_examples'] + dimension['high_examples']
        probing = generate_probing_texts(all_prompts, adapter)
        all_probing.extend(probing)
    
    # Save activation pairs for operator learning
    pairs_file = output_dir / f"activation_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with jsonlines.open(pairs_file, 'w') as writer:
        writer.write_all(all_pairs)
    print(f"\n✓ Saved {len(all_pairs)} activation pairs to {pairs_file}")
    
    # Save probing data for testing
    probing_file = output_dir / f"probing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with jsonlines.open(probing_file, 'w') as writer:
        writer.write_all(all_probing)
    print(f"✓ Saved {len(all_probing)} probing samples to {probing_file}")
    
    # Generate metrics
    metrics = {
        'model': model_name,
        'device': str(device),
        'hidden_dim': adapter.get_hidden_dim(),
        'extraction_layers': config['experiment']['extraction_layers'],
        'num_dimensions': len(config['experiment']['dimensions']),
        'total_pairs': len(all_pairs),
        'total_probing': len(all_probing),
        'pairs_file': str(pairs_file),
        'probing_file': str(probing_file),
        'timestamp': datetime.now().isoformat(),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'not_set')
    }
    
    # Save metrics
    results_dir = Path(config['paths'][paths_mode]['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "stage1_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Stage 1 complete!")
    print(f"✓ Model: {model_name} ({adapter.get_hidden_dim()}D)")
    print(f"✓ Extracted real activations from layers {config['experiment']['extraction_layers']}")
    print(f"✓ Ready for HDC projection and operator learning")
    
    # Quick test: show that activations differ between low/high certainty
    if all_pairs:
        sample_pair = all_pairs[0]
        low_act = np.array(sample_pair['low_activation'])
        high_act = np.array(sample_pair['high_activation'])
        
        # Compute basic statistics
        diff = high_act - low_act
        magnitude = np.linalg.norm(diff.flatten())
        
        print(f"\n📊 Sample activation difference ({sample_pair['dimension']}):")
        print(f"   Low text: '{sample_pair['low_text']}'")
        print(f"   High text: '{sample_pair['high_text']}'")
        print(f"   Activation difference magnitude: {magnitude:.4f}")
        print(f"   This difference is what operators will learn to induce!")


if __name__ == "__main__":
    main()