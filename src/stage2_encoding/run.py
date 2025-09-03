#!/usr/bin/env python3
"""
Stage 2: HDC Projection and Encoding
Projects real activations to HDC space with multiple strategies to test H3 hypothesis.
Supports random, learned, and padding projections.
"""

import os
os.environ['PYTHONHASHSEED'] = '42'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import jsonlines
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import yaml
from tqdm import tqdm

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
    return {}


class ProjectionStrategy:
    """Base class for projection strategies"""
    
    def __init__(self, input_dim: int, output_dim: int = 10000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def project(self, activation: torch.Tensor) -> torch.Tensor:
        """Project activation to HDC space"""
        raise NotImplementedError
        
    def inverse_project(self, cogit: torch.Tensor) -> torch.Tensor:
        """Project cogit back to activation space"""
        raise NotImplementedError


class RandomProjection(ProjectionStrategy):
    """Random projection using fixed random matrix"""
    
    def __init__(self, input_dim: int, output_dim: int = 10000):
        super().__init__(input_dim, output_dim)
        # Create fixed random projection matrices
        torch.manual_seed(42)
        self.projection_matrix = torch.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.inverse_matrix = torch.randn(output_dim, input_dim) / np.sqrt(output_dim)
        
    def project(self, activation: torch.Tensor) -> torch.Tensor:
        """Random projection to HDC space"""
        # Handle batch and sequence dimensions
        original_shape = activation.shape
        if len(original_shape) > 2:
            # Flatten all but last dimension
            activation = activation.view(-1, self.input_dim)
        
        # Project
        cogit = torch.matmul(activation, self.projection_matrix)
        
        # Normalize for HDC
        cogit = torch.tanh(cogit)  # Bound to [-1, 1]
        
        return cogit
        
    def inverse_project(self, cogit: torch.Tensor) -> torch.Tensor:
        """Random projection back to activation space"""
        return torch.matmul(cogit, self.inverse_matrix)


class LearnedProjection(ProjectionStrategy):
    """Learned projection using autoencoder approach"""
    
    def __init__(self, input_dim: int, output_dim: int = 10000):
        super().__init__(input_dim, output_dim)
        
        # Build encoder and decoder networks
        hidden_dim = (input_dim + output_dim) // 2
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Bound outputs for HDC
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.is_trained = False
        
    def train_projection(self, activation_data: List[torch.Tensor], epochs: int = 100):
        """Train the projection to preserve information"""
        print("Training learned projection...")
        
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.001
        )
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for activation in activation_data:
                # Handle different shapes
                if len(activation.shape) > 2:
                    activation = activation.view(-1, self.input_dim)
                
                # Forward pass
                cogit = self.encoder(activation)
                reconstructed = self.decoder(cogit)
                
                # Compute loss
                loss = criterion(reconstructed, activation)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(activation_data)
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        print("✓ Learned projection trained")
        
    def project(self, activation: torch.Tensor) -> torch.Tensor:
        """Learned projection to HDC space"""
        if not self.is_trained:
            print("Warning: Using untrained learned projection")
        
        # Handle batch and sequence dimensions
        original_shape = activation.shape
        if len(original_shape) > 2:
            activation = activation.view(-1, self.input_dim)
        
        with torch.no_grad():
            cogit = self.encoder(activation)
        
        return cogit
        
    def inverse_project(self, cogit: torch.Tensor) -> torch.Tensor:
        """Learned projection back to activation space"""
        with torch.no_grad():
            return self.decoder(cogit)


class PaddingProjection(ProjectionStrategy):
    """Simple padding/truncation projection"""
    
    def __init__(self, input_dim: int, output_dim: int = 10000):
        super().__init__(input_dim, output_dim)
        
    def project(self, activation: torch.Tensor) -> torch.Tensor:
        """Pad or truncate to reach HDC dimension"""
        # Handle batch and sequence dimensions
        original_shape = activation.shape
        if len(original_shape) > 2:
            activation = activation.view(-1, self.input_dim)
        
        if self.input_dim < self.output_dim:
            # Pad with zeros
            padding = torch.zeros(activation.shape[0], self.output_dim - self.input_dim)
            cogit = torch.cat([activation, padding], dim=-1)
        else:
            # Truncate
            cogit = activation[:, :self.output_dim]
        
        # Normalize for HDC
        cogit = torch.tanh(cogit)
        
        return cogit
        
    def inverse_project(self, cogit: torch.Tensor) -> torch.Tensor:
        """Inverse padding/truncation"""
        if self.input_dim < self.output_dim:
            # Remove padding
            return cogit[:, :self.input_dim]
        else:
            # Pad back (information lost)
            padding = torch.zeros(cogit.shape[0], self.input_dim - self.output_dim)
            return torch.cat([cogit, padding], dim=-1)


class HDCEncoder:
    """Encodes activations into HDC cogit vectors using specified projection strategy"""
    
    def __init__(self, projection_strategy: ProjectionStrategy, hdc_dim: int = 10000):
        self.projection = projection_strategy
        self.hdc_dim = hdc_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def encode_activation(self, activation: np.ndarray) -> torch.Tensor:
        """Convert activation to cogit hypervector"""
        # Convert to tensor
        if isinstance(activation, np.ndarray):
            activation = torch.tensor(activation, dtype=torch.float32)
        
        # Project to HDC space
        cogit = self.projection.project(activation)
        
        return cogit
        
    def decode_cogit(self, cogit: torch.Tensor) -> torch.Tensor:
        """Convert cogit back to activation space"""
        return self.projection.inverse_project(cogit)


def process_activation_pairs(pairs_file: Path, strategy_name: str, input_dim: int) -> Dict[str, Any]:
    """Process activation pairs with specified projection strategy"""
    print(f"\n🔄 Processing with {strategy_name} projection...")
    
    # Create projection strategy
    if strategy_name == "random":
        projection = RandomProjection(input_dim)
    elif strategy_name == "learned":
        projection = LearnedProjection(input_dim)
    elif strategy_name == "padding":
        projection = PaddingProjection(input_dim)
    else:
        raise ValueError(f"Unknown projection strategy: {strategy_name}")
    
    # Load activation pairs
    pairs = []
    with jsonlines.open(pairs_file) as reader:
        pairs = list(reader)
    
    # If using learned projection, train it first
    if strategy_name == "learned" and pairs:
        # Collect all activations for training
        all_activations = []
        for pair in pairs[:100]:  # Use subset for training
            low_act = torch.tensor(pair['low_activation'], dtype=torch.float32)
            high_act = torch.tensor(pair['high_activation'], dtype=torch.float32)
            all_activations.extend([low_act, high_act])
        
        # Train the projection
        projection.train_projection(all_activations, epochs=50)
    
    # Create encoder
    encoder = HDCEncoder(projection)
    
    # Process pairs
    encoded_pairs = []
    reconstruction_errors = []
    
    for pair in tqdm(pairs, desc=f"Encoding with {strategy_name}"):
        # Encode low and high activations
        low_act = np.array(pair['low_activation'])
        high_act = np.array(pair['high_activation'])
        
        low_cogit = encoder.encode_activation(low_act)
        high_cogit = encoder.encode_activation(high_act)
        
        # Test reconstruction quality
        low_reconstructed = encoder.decode_cogit(low_cogit)
        high_reconstructed = encoder.decode_cogit(high_cogit)
        
        # Flatten for comparison
        low_act_flat = torch.tensor(low_act).flatten()
        high_act_flat = torch.tensor(high_act).flatten()
        
        # Ensure dimensions match for error calculation
        min_dim = min(low_act_flat.shape[0], low_reconstructed.shape[0])
        low_error = torch.mean((low_act_flat[:min_dim] - low_reconstructed.flatten()[:min_dim]) ** 2).item()
        high_error = torch.mean((high_act_flat[:min_dim] - high_reconstructed.flatten()[:min_dim]) ** 2).item()
        
        reconstruction_errors.append((low_error + high_error) / 2)
        
        # Store encoded pair
        encoded_pair = {
            'dimension': pair['dimension'],
            'layer': pair['layer'],
            'low_text': pair['low_text'],
            'high_text': pair['high_text'],
            'low_cogit': low_cogit.cpu().numpy().tolist(),
            'high_cogit': high_cogit.cpu().numpy().tolist(),
            'projection_strategy': strategy_name,
            'reconstruction_error': (low_error + high_error) / 2,
            'timestamp': datetime.now().isoformat()
        }
        encoded_pairs.append(encoded_pair)
    
    # Calculate statistics
    avg_reconstruction_error = np.mean(reconstruction_errors)
    
    return {
        'encoded_pairs': encoded_pairs,
        'avg_reconstruction_error': avg_reconstruction_error,
        'projection': projection
    }


def main():
    """Main encoding pipeline with multiple projection strategies"""
    print("🔢 Stage 2: HDC Projection and Encoding Pipeline")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    model_name = config['model']['name']
    model_config = config['model']['configs'].get(model_name, {})
    input_dim = model_config.get('hidden_dim', 768)
    
    paths_mode = config['paths']['mode']
    base_path = config['paths'][paths_mode]['data_dir']
    
    print(f"Model: {model_name}")
    print(f"Input dimension: {input_dim}")
    print(f"HDC dimension: {config['hdc']['dimension']}")
    
    # Find latest activation pairs file
    input_dir = Path(base_path) / "raw" / "activations"
    pairs_files = list(input_dir.glob("activation_pairs_*.jsonl"))
    
    if not pairs_files:
        raise FileNotFoundError("No activation pairs found. Run Stage 1 first.")
    
    latest_pairs = sorted(pairs_files)[-1]
    print(f"Using activation file: {latest_pairs.name}")
    
    # Create output directory
    output_dir = Path(base_path) / "processed" / "cogits"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each projection strategy
    projection_strategies = config['projections']['strategies']
    results = {}
    
    for strategy in projection_strategies:
        result = process_activation_pairs(latest_pairs, strategy, input_dim)
        results[strategy] = result
        
        # Save encoded pairs
        output_file = output_dir / f"cogits_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(result['encoded_pairs'])
        
        print(f"✓ {strategy}: {len(result['encoded_pairs'])} pairs encoded")
        print(f"  Avg reconstruction error: {result['avg_reconstruction_error']:.6f}")
        print(f"  Saved to: {output_file.name}")
    
    # Compare strategies
    print("\n📊 Projection Strategy Comparison:")
    print("-" * 40)
    for strategy, result in results.items():
        print(f"{strategy:12s}: reconstruction error = {result['avg_reconstruction_error']:.6f}")
    
    # Find best strategy (lowest reconstruction error)
    best_strategy = min(results.items(), key=lambda x: x[1]['avg_reconstruction_error'])[0]
    print(f"\n🏆 Best strategy: {best_strategy}")
    
    # Generate metrics
    metrics = {
        'model': model_name,
        'input_dim': input_dim,
        'hdc_dim': config['hdc']['dimension'],
        'projection_strategies': projection_strategies,
        'strategy_results': {
            strategy: {
                'num_pairs': len(result['encoded_pairs']),
                'avg_reconstruction_error': result['avg_reconstruction_error']
            }
            for strategy, result in results.items()
        },
        'best_strategy': best_strategy,
        'timestamp': datetime.now().isoformat(),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'not_set')
    }
    
    # Save metrics
    results_dir = Path(config['paths'][paths_mode]['results_dir'])
    with open(results_dir / "stage2_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Stage 2 complete!")
    print(f"✓ Tested {len(projection_strategies)} projection strategies")
    print(f"✓ {best_strategy} projection shows best reconstruction")
    print(f"✓ Ready for operator learning in Stage 3")


if __name__ == "__main__":
    main()