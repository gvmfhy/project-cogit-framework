#!/usr/bin/env python3
"""
Stage 3: Learning - Fit operator models on cogit pairs/sequences
Includes inverse/no-op checks and metrics logged to /results for DVC tracking.
"""

import os
# Set PYTHONHASHSEED for deterministic execution
os.environ['PYTHONHASHSEED'] = '42'

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
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

class CogitOperator(nn.Module):
    """
    Neural network that learns to transform cogit hypervectors.
    Implements operators that can manipulate cognitive states over time.
    """
    
    def __init__(self, vector_dim: int, hidden_dim: int = 512, operator_type: str = 'linear'):
        super().__init__()
        self.vector_dim = vector_dim
        self.operator_type = operator_type
        
        if operator_type == 'linear':
            # Simple linear transformation
            self.operator = nn.Linear(vector_dim, vector_dim)
        elif operator_type == 'mlp':
            # Multi-layer perceptron
            self.operator = nn.Sequential(
                nn.Linear(vector_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, vector_dim)
            )
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")
    
    def forward(self, cogit: torch.Tensor) -> torch.Tensor:
        """Apply learned transformation to cogit vector"""
        return self.operator(cogit)

class CogitOperatorTrainer:
    """Trains cogit operators with inverse/no-op sanity checks"""
    
    def __init__(self, vector_dim: int, device: torch.device):
        self.vector_dim = vector_dim
        self.device = device
        
    def create_training_pairs(self, cogits: List[Dict[str, Any]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create training pairs from cogit sequences.
        Uses consecutive cogits from same conversation as input-output pairs.
        """
        pairs = []
        
        # Group by conversation
        conversations = {}
        for cogit_data in cogits:
            conv_id = cogit_data['conversation_id']
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(cogit_data)
        
        # Create consecutive pairs within conversations
        for conv_id, conv_cogits in conversations.items():
            # Sort by turn number
            conv_cogits.sort(key=lambda x: x['turn'])
            
            for i in range(len(conv_cogits) - 1):
                input_cogit = torch.tensor(conv_cogits[i]['cogit_vector'], dtype=torch.float32)
                target_cogit = torch.tensor(conv_cogits[i + 1]['cogit_vector'], dtype=torch.float32)
                
                pairs.append((input_cogit, target_cogit))
        
        return pairs
    
    def train_operator(self, operator: CogitOperator, training_pairs: List[Tuple[torch.Tensor, torch.Tensor]], 
                      epochs: int, learning_rate: float) -> Dict[str, List[float]]:
        """Train the cogit operator"""
        operator.to(self.device)
        optimizer = optim.Adam(operator.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        history = {'loss': [], 'epoch': []}
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle training pairs
            random.shuffle(training_pairs)
            
            for input_cogit, target_cogit in training_pairs:
                input_cogit = input_cogit.to(self.device)
                target_cogit = target_cogit.to(self.device)
                
                # Forward pass
                predicted_cogit = operator(input_cogit)
                loss = criterion(predicted_cogit, target_cogit)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            history['epoch'].append(epoch + 1)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return history
    
    def test_inverse_property(self, operator: CogitOperator, test_cogits: List[torch.Tensor]) -> Dict[str, float]:
        """
        Test inverse property: operator(operator(x)) should approximate x for some operators.
        This is a sanity check for operator consistency.
        """
        operator.eval()
        mse_errors = []
        cosine_similarities = []
        
        with torch.no_grad():
            for cogit in test_cogits[:min(100, len(test_cogits))]:  # Test on subset
                cogit = cogit.to(self.device)
                
                # Apply operator twice
                transformed = operator(cogit)
                double_transformed = operator(transformed)
                
                # Calculate reconstruction error
                mse = torch.mean((cogit - double_transformed) ** 2).item()
                
                # Calculate cosine similarity
                cos_sim = torch.cosine_similarity(cogit.unsqueeze(0), double_transformed.unsqueeze(0)).item()
                
                mse_errors.append(mse)
                cosine_similarities.append(cos_sim)
        
        return {
            'avg_mse_error': np.mean(mse_errors),
            'avg_cosine_similarity': np.mean(cosine_similarities),
            'num_tests': len(mse_errors)
        }
    
    def test_no_op_baseline(self, test_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """
        Test no-op baseline: what would performance be if operator did nothing?
        This provides a baseline for operator effectiveness.
        """
        mse_errors = []
        cosine_similarities = []
        
        for input_cogit, target_cogit in test_pairs[:min(100, len(test_pairs))]:
            # No-op: input equals output
            mse = torch.mean((input_cogit - target_cogit) ** 2).item()
            cos_sim = torch.cosine_similarity(input_cogit.unsqueeze(0), target_cogit.unsqueeze(0)).item()
            
            mse_errors.append(mse)
            cosine_similarities.append(cos_sim)
        
        return {
            'no_op_mse': np.mean(mse_errors),
            'no_op_cosine_similarity': np.mean(cosine_similarities),
            'num_tests': len(mse_errors)
        }

def load_params() -> Dict[str, Any]:
    """Load learning parameters from params.yaml"""
    params_file = Path("params.yaml")
    if params_file.exists():
        with open(params_file) as f:
            params = yaml.safe_load(f)
            return params.get('learn', {})
    
    # Default parameters
    return {
        'model_type': 'linear',  # 'linear' or 'mlp'
        'epochs': 50,
        'learning_rate': 0.001,
        'test_split': 0.2
    }

def load_cogits(data_dir: Path) -> List[Dict[str, Any]]:
    """Load encoded cogits from .pt files"""
    cogits = []
    
    for pt_file in data_dir.glob("*.pt"):
        data = torch.load(pt_file, map_location='cpu')
        if 'cogits' in data:
            cogits.extend(data['cogits'])
    
    return cogits

def main():
    """Main learning pipeline"""
    print("🧠 Stage 3: Starting Learning Pipeline")
    print("=" * 50)
    
    # Load parameters
    params = load_params()
    print(f"Parameters: {params}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load cogit data
    data_dir = Path("data/processed/cogits")
    if not data_dir.exists():
        raise FileNotFoundError(f"Input directory {data_dir} does not exist. Run Stage 2 first.")
    
    cogits = load_cogits(data_dir)
    print(f"Loaded {len(cogits)} encoded cogits")
    
    if len(cogits) == 0:
        raise ValueError("No cogits found. Ensure Stage 2 has been run successfully.")
    
    # Determine vector dimension from first cogit
    vector_dim = len(cogits[0]['cogit_vector'])
    print(f"Cogit vector dimension: {vector_dim}")
    
    # Initialize trainer
    trainer = CogitOperatorTrainer(vector_dim, device)
    
    # Create training pairs
    training_pairs = trainer.create_training_pairs(cogits)
    print(f"Created {len(training_pairs)} training pairs")
    
    if len(training_pairs) == 0:
        raise ValueError("No training pairs created. Need multiple turns per conversation.")
    
    # Split into train/test
    test_size = int(len(training_pairs) * params['test_split'])
    random.shuffle(training_pairs)
    test_pairs = training_pairs[:test_size]
    train_pairs = training_pairs[test_size:]
    
    print(f"Training pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
    
    # Create and train operator
    operator = CogitOperator(vector_dim, operator_type=params['model_type'])
    print(f"Created {params['model_type']} operator with {sum(p.numel() for p in operator.parameters())} parameters")
    
    # Train the operator
    print("\nTraining operator...")
    history = trainer.train_operator(
        operator, 
        train_pairs, 
        params['epochs'], 
        params['learning_rate']
    )
    
    # Perform sanity checks
    print("\nPerforming sanity checks...")
    
    # Test inverse property
    test_cogits = [torch.tensor(cogit['cogit_vector'], dtype=torch.float32) for cogit in cogits[:100]]
    inverse_results = trainer.test_inverse_property(operator, test_cogits)
    
    # Test no-op baseline
    no_op_results = trainer.test_no_op_baseline(test_pairs)
    
    # Create output directories
    models_dir = Path("models/operators")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trained model
    model_file = models_dir / f"cogit_operator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': operator.state_dict(),
        'model_type': params['model_type'],
        'vector_dim': vector_dim,
        'training_history': history,
        'training_params': params,
        'inverse_test_results': inverse_results,
        'no_op_baseline': no_op_results,
        'metadata': {
            'num_training_pairs': len(train_pairs),
            'num_test_pairs': len(test_pairs),
            'timestamp': datetime.now().isoformat(),
            'seed': 42
        }
    }, model_file)
    
    # Generate comprehensive metrics
    final_loss = history['loss'][-1] if history['loss'] else float('inf')
    
    metrics = {
        'training_metrics': {
            'final_loss': final_loss,
            'num_epochs': params['epochs'],
            'learning_rate': params['learning_rate'],
            'num_parameters': sum(p.numel() for p in operator.parameters()),
            'training_pairs': len(train_pairs),
            'test_pairs': len(test_pairs)
        },
        'sanity_checks': {
            'inverse_property': inverse_results,
            'no_op_baseline': no_op_results,
            'operator_effectiveness': {
                'better_than_no_op': final_loss < no_op_results['no_op_mse'],
                'loss_improvement': no_op_results['no_op_mse'] - final_loss
            }
        },
        'model_info': {
            'model_type': params['model_type'],
            'vector_dimension': vector_dim,
            'model_file': str(model_file),
            'deterministic_training': True
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'seed': 42,
            'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'not_set'),
            'device': str(device)
        }
    }
    
    # Save metrics
    metrics_file = results_dir / f"learn_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also save to DVC expected location
    with open("results/learn_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Learning complete!")
    print(f"✓ Model saved to {model_file}")
    print(f"✓ Metrics saved to {metrics_file}")
    print(f"✓ Final training loss: {final_loss:.6f}")
    print(f"✓ No-op baseline MSE: {no_op_results['no_op_mse']:.6f}")
    print(f"✓ Inverse test MSE: {inverse_results['avg_mse_error']:.6f}")
    print(f"✓ Inverse test cosine similarity: {inverse_results['avg_cosine_similarity']:.4f}")
    print(f"✓ PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")

if __name__ == "__main__":
    main()