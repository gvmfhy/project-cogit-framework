#!/usr/bin/env python3
"""
Stage 2: Encoding - Transform per-turn states into HDC hypervectors
Uses TorchHD/hdlib semantics with clear binding/bundling/permutation ops and fixed seeds.
"""

import os
# Set PYTHONHASHSEED for deterministic execution
os.environ['PYTHONHASHSEED'] = '42'

import random
import numpy as np
import torch
import json
import jsonlines
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import yaml

# Import TorchHD for hyperdimensional computing
try:
    import torch_hd
    HDC_AVAILABLE = True
except ImportError:
    print("Warning: TorchHD not available. Using mock HDC operations.")
    HDC_AVAILABLE = False

# Deterministic seeding
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

class CogitEncoder:
    """
    Encodes cognitive states into hyperdimensional cogit vectors.
    Implements HDC operations: bind ≈ XOR or circular convolution, bundle ≈ (normalized) sum, permute ≈ fixed permutation
    """
    
    def __init__(self, dim: int = 10000, model: str = 'binary'):
        self.dim = dim
        self.model = model  # 'binary' or 'real'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate deterministic basis vectors for cognitive dimensions
        torch.manual_seed(42)  # Ensure reproducible basis vectors
        self.basis_vectors = self._create_basis_vectors()
        self.permutations = self._create_permutations()
        
    def _create_basis_vectors(self) -> Dict[str, torch.Tensor]:
        """Create random basis vectors for each cognitive dimension"""
        dimensions = ['agreement', 'certainty', 'openness', 'emotional_tone', 'social_alignment']
        basis = {}
        
        for dim_name in dimensions:
            if self.model == 'binary':
                # Binary hypervectors: {-1, +1}
                vec = torch.randint(0, 2, (self.dim,), device=self.device) * 2 - 1
            else:
                # Real hypervectors: normalized Gaussian
                vec = torch.randn(self.dim, device=self.device)
                vec = vec / torch.norm(vec)  # Normalize to unit length
            
            basis[dim_name] = vec.float()
            
        return basis
    
    def _create_permutations(self) -> Dict[str, torch.Tensor]:
        """Create fixed permutation indices for each cognitive dimension"""
        dimensions = list(self.basis_vectors.keys())
        permutations = {}
        
        for i, dim_name in enumerate(dimensions):
            # Create deterministic permutation based on dimension index
            torch.manual_seed(42 + i)  
            perm = torch.randperm(self.dim, device=self.device)
            permutations[dim_name] = perm
            
        return permutations
    
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        HDC bind operation ≈ XOR or circular convolution (per chosen VSA model)
        Property: Approximate inverse a ⊛ b ⊛ b ≈ a
        """
        if HDC_AVAILABLE:
            return torch_hd.bind(a, b)
        else:
            # Mock implementation: XOR for binary, elementwise multiplication for real
            if self.model == 'binary':
                return a * b  # XOR equivalent for {-1, +1} vectors
            else:
                return a * b  # Elementwise product approximation
    
    def bundle(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        HDC bundle operation ≈ (normalized) sum
        Property: Similarity preservation sim(a ⊕ b, a) > threshold
        """
        if HDC_AVAILABLE:
            return torch_hd.bundle(vectors)
        else:
            # Mock implementation: sum and normalize/sign
            bundled = torch.stack(vectors).sum(dim=0)
            
            if self.model == 'binary':
                return torch.sign(bundled)  # Majority rule for binary vectors
            else:
                return bundled / torch.norm(bundled)  # Normalized sum for real vectors
    
    def permute(self, vector: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        """
        HDC permute operation ≈ fixed permutation
        Property: Invertible π⁻¹(π(a)) = a
        """
        if HDC_AVAILABLE:
            return torch_hd.permute(vector, permutation)
        else:
            # Mock implementation: apply permutation indices
            return vector[permutation]
    
    def encode_cognitive_state(self, state: Dict[str, float]) -> torch.Tensor:
        """
        Encode cognitive state into cogit hypervector using HDC operations.
        Formula: cogit = π₁(certainty ⊛ c_basis) ⊕ π₂(agreement ⊛ a_basis) ⊕ ...
        """
        encoded_dims = []
        
        for dim_name, value in state.items():
            if dim_name not in self.basis_vectors:
                continue
                
            # Convert scalar value to hypervector (simple scaling)
            if self.model == 'binary':
                value_vec = torch.full((self.dim,), value, device=self.device)
                value_vec = torch.sign(value_vec)  # Binarize
            else:
                value_vec = torch.full((self.dim,), value, device=self.device)
            
            # Bind value with basis vector: value ⊛ basis
            bound = self.bind(value_vec, self.basis_vectors[dim_name])
            
            # Apply dimension-specific permutation: π(bound)
            permuted = self.permute(bound, self.permutations[dim_name])
            
            encoded_dims.append(permuted)
        
        # Bundle all dimensions: ⊕ all encoded dimensions
        if encoded_dims:
            cogit = self.bundle(encoded_dims)
        else:
            # Fallback: zero vector
            cogit = torch.zeros(self.dim, device=self.device)
        
        return cogit

def load_params() -> Dict[str, Any]:
    """Load encoding parameters from params.yaml"""
    params_file = Path("params.yaml")
    if params_file.exists():
        with open(params_file) as f:
            params = yaml.safe_load(f)
            return params.get('encode', {})
    
    # Default parameters
    return {
        'vector_dim': 10000,
        'hdc_model': 'binary',  # 'binary' or 'real'
        'seed': 42
    }

def load_conversations(data_dir: Path) -> List[Dict[str, Any]]:
    """Load conversation data from JSONL files"""
    conversations = []
    
    for jsonl_file in data_dir.glob("*.jsonl"):
        with jsonlines.open(jsonl_file) as reader:
            conversations.extend(list(reader))
    
    return conversations

def main():
    """Main encoding pipeline"""
    print("🔢 Stage 2: Starting Encoding Pipeline")
    print("=" * 50)
    
    # Load parameters
    params = load_params()
    print(f"Parameters: {params}")
    
    # Set additional seeds from parameters
    if 'seed' in params:
        torch.manual_seed(params['seed'])
        random.seed(params['seed'])
        np.random.seed(params['seed'])
    
    # Load conversation data
    data_dir = Path("data/raw/sims")
    if not data_dir.exists():
        raise FileNotFoundError(f"Input directory {data_dir} does not exist. Run Stage 1 first.")
    
    conversations = load_conversations(data_dir)
    print(f"Loaded {len(conversations)} conversation turns")
    
    # Initialize encoder
    encoder = CogitEncoder(
        dim=params['vector_dim'],
        model=params['hdc_model']
    )
    
    # Create output directory
    output_dir = Path("data/processed/cogits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Encode cognitive states
    encoded_cogits = []
    
    for turn_data in conversations:
        cognitive_state = turn_data['cognitive_state']
        
        # Encode to cogit hypervector
        cogit = encoder.encode_cognitive_state(cognitive_state)
        
        # Prepare output data
        cogit_data = {
            'conversation_id': turn_data['conversation_id'],
            'turn': turn_data['turn'],
            'speaker': turn_data['speaker'],
            'topic': turn_data['topic'],
            'original_state': cognitive_state,
            'cogit_vector': cogit.cpu().numpy().tolist(),  # Convert to list for JSON serialization
            'timestamp': datetime.now().isoformat()
        }
        
        encoded_cogits.append(cogit_data)
    
    # Save encoded cogits
    output_file = output_dir / f"cogits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'cogits': encoded_cogits,
        'encoder_params': {
            'dim': params['vector_dim'],
            'model': params['hdc_model'],
            'basis_vectors': {k: v.cpu() for k, v in encoder.basis_vectors.items()},
            'permutations': {k: v.cpu() for k, v in encoder.permutations.items()}
        },
        'metadata': {
            'num_cogits': len(encoded_cogits),
            'hdc_library': 'torchhd' if HDC_AVAILABLE else 'mock',
            'timestamp': datetime.now().isoformat(),
            'seed': params['seed']
        }
    }, output_file)
    
    # Generate metrics
    metrics = {
        'total_cogits_encoded': len(encoded_cogits),
        'vector_dimension': params['vector_dim'],
        'hdc_model': params['hdc_model'],
        'hdc_library_available': HDC_AVAILABLE,
        'output_file': str(output_file),
        'deterministic_encoding': True,
        'timestamp': datetime.now().isoformat(),
        'seed': params['seed'],
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'not_set')
    }
    
    # Save metrics
    metrics_dir = Path("results")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "encode_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Encoding complete: {len(encoded_cogits)} cogits saved to {output_file}")
    print(f"✓ Metrics saved to results/encode_metrics.json")
    print(f"✓ HDC Library Available: {HDC_AVAILABLE}")
    print(f"✓ PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")

if __name__ == "__main__":
    main()