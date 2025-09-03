#!/usr/bin/env python3
"""
Stage 2: HDC Encoding with Clean Data Flow
Transforms activation pairs into hyperdimensional cognitive vectors (cogits).
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import random
import numpy as np
import torch
import torch.nn as nn
import json
import jsonlines
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import yaml

# Deterministic seeding
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class HDCEncoder:
    """Encode activation vectors into hyperdimensional cognitive vectors (cogits)"""
    
    def __init__(self, input_dim: int = 768, hd_dim: int = 10000):
        self.input_dim = input_dim
        self.hd_dim = hd_dim
        
        # Create random projection matrix (deterministic with seed)
        self.projection = self._create_projection_matrix()
        
        # Create basis vectors for each dimension
        self.basis_vectors = {}
        
    def _create_projection_matrix(self) -> torch.Tensor:
        """Create a random projection matrix for encoding"""
        # Random projection from input_dim to hd_dim
        matrix = torch.randn(self.input_dim, self.hd_dim)
        # Normalize columns
        matrix = matrix / torch.norm(matrix, dim=0, keepdim=True)
        return matrix
    
    def _get_or_create_basis(self, dimension_name: str) -> torch.Tensor:
        """Get or create a basis vector for a cognitive dimension"""
        if dimension_name not in self.basis_vectors:
            # Create random hypervector as basis
            basis = torch.randn(self.hd_dim)
            basis = torch.sign(basis)  # Make it binary (-1, +1)
            self.basis_vectors[dimension_name] = basis
        return self.basis_vectors[dimension_name]
    
    def encode_activation(self, activation: np.ndarray, dimension: str) -> torch.Tensor:
        """
        Encode an activation vector into a hyperdimensional cogit.
        
        Formula: cogit = project(activation) ⊛ basis_vector
        Where ⊛ is the binding operation (element-wise multiplication for simplicity)
        """
        # Convert to tensor
        act_tensor = torch.tensor(activation, dtype=torch.float32)
        
        # Handle different shapes (flatten if needed)
        if len(act_tensor.shape) > 1:
            act_tensor = act_tensor.flatten()
        
        # Ensure correct dimensionality
        if act_tensor.shape[0] != self.input_dim:
            # Pad or truncate as needed
            if act_tensor.shape[0] < self.input_dim:
                padding = torch.zeros(self.input_dim - act_tensor.shape[0])
                act_tensor = torch.cat([act_tensor, padding])
            else:
                act_tensor = act_tensor[:self.input_dim]
        
        # Project to HD space
        hd_vector = torch.matmul(act_tensor, self.projection)
        
        # Normalize and binarize
        hd_vector = torch.sign(hd_vector)
        
        # Bind with dimension basis
        basis = self._get_or_create_basis(dimension)
        cogit = hd_vector * basis  # Element-wise multiplication (binding)
        
        return cogit
    
    def encode_pair(self, pair_data: Dict) -> Dict[str, Any]:
        """Encode a pair of activations into cogits"""
        dimension = pair_data['dimension']
        layer = pair_data['layer']
        
        # Encode low and high activations
        low_act = np.array(pair_data['low_activation'])
        high_act = np.array(pair_data['high_activation'])
        
        low_cogit = self.encode_activation(low_act, dimension)
        high_cogit = self.encode_activation(high_act, dimension)
        
        # Calculate similarity for validation
        similarity = torch.cosine_similarity(low_cogit.unsqueeze(0), 
                                            high_cogit.unsqueeze(0)).item()
        
        return {
            'dimension': dimension,
            'layer': layer,
            'low_text': pair_data['low_text'],
            'high_text': pair_data['high_text'],
            'low_cogit': low_cogit.numpy().tolist(),
            'high_cogit': high_cogit.numpy().tolist(),
            'similarity': similarity,
            'timestamp': datetime.now().isoformat()
        }


def process_activations(input_file: Path, output_dir: Path) -> Path:
    """Process activation pairs into cogits"""
    print("\n" + "=" * 60)
    print("STAGE 2: HDC ENCODING")
    print("=" * 60)
    
    # Initialize encoder
    encoder = HDCEncoder(input_dim=768, hd_dim=10000)
    print(f"\n✓ Initialized HDC encoder (768 → 10000 dimensions)")
    
    # Load activation pairs
    pairs = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            pairs.append(obj)
    print(f"✓ Loaded {len(pairs)} activation pairs")
    
    # Encode all pairs
    encoded_pairs = []
    dimensions_seen = set()
    
    print("\nEncoding pairs into cogits...")
    for i, pair in enumerate(pairs):
        encoded = encoder.encode_pair(pair)
        encoded_pairs.append(encoded)
        dimensions_seen.add(pair['dimension'])
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs...")
    
    print(f"\n✓ Encoded {len(encoded_pairs)} pairs")
    print(f"✓ Dimensions processed: {', '.join(dimensions_seen)}")
    
    # Calculate statistics
    similarities = [p['similarity'] for p in encoded_pairs]
    print(f"\nCogit similarity statistics:")
    print(f"  Mean: {np.mean(similarities):.4f}")
    print(f"  Std:  {np.std(similarities):.4f}")
    print(f"  Min:  {np.min(similarities):.4f}")
    print(f"  Max:  {np.max(similarities):.4f}")
    
    # Save encoded cogits
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cogits_{timestamp}.jsonl"
    
    with jsonlines.open(output_file, 'w') as writer:
        for encoded in encoded_pairs:
            writer.write(encoded)
    
    print(f"\n✓ Saved cogits to {output_file}")
    
    # Save encoding metadata
    metadata = {
        'input_file': str(input_file),
        'num_pairs': len(encoded_pairs),
        'dimensions': list(dimensions_seen),
        'hd_dim': encoder.hd_dim,
        'input_dim': encoder.input_dim,
        'timestamp': timestamp
    }
    
    metadata_file = output_dir / f"encoding_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_file}")
    
    return output_file


def run_encoding_pipeline():
    """Run the full encoding pipeline"""
    # Find the latest activation file
    sim_dir = Path("data/raw/sims")
    activation_files = list(sim_dir.glob("activations_*.jsonl"))
    
    if not activation_files:
        print("No activation files found! Run Stage 1 first.")
        return None
    
    # Use the most recent file
    latest_file = max(activation_files, key=lambda p: p.stat().st_mtime)
    print(f"Using activation file: {latest_file}")
    
    # Process into cogits
    output_dir = Path("data/processed/cogits")
    output_file = process_activations(latest_file, output_dir)
    
    return output_file


if __name__ == "__main__":
    output_file = run_encoding_pipeline()
    
    if output_file:
        print("\n" + "=" * 60)
        print("Stage 2 Complete - Ready for Stage 3 (Learning)")
        print("=" * 60)