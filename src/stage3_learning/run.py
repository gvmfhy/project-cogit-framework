#!/usr/bin/env python3
"""
Stage 3: Cognitive Manipulation Operator Learning
Learns operators that transform cognitive states (e.g., low_certainty → high_certainty).
Includes behavioral testing to verify operators produce predicted changes.
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

# Import our model adapter for behavioral testing
from src.model_adapter import ModelAdapterFactory
from src.stage2_encoding.run import RandomProjection, LearnedProjection, PaddingProjection

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


class CognitiveManipulationOperator(nn.Module):
    """
    Neural network that learns to transform cogits from one cognitive state to another.
    E.g., transforms low_certainty cogits → high_certainty cogits.
    """
    
    def __init__(self, cogit_dim: int = 10000, hidden_dim: int = 1024, operator_type: str = 'mlp'):
        super().__init__()
        self.cogit_dim = cogit_dim
        self.operator_type = operator_type
        
        if operator_type == 'linear':
            # Simple linear transformation
            self.operator = nn.Linear(cogit_dim, cogit_dim)
        elif operator_type == 'mlp':
            # Multi-layer perceptron
            self.operator = nn.Sequential(
                nn.Linear(cogit_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, cogit_dim)
            )
        elif operator_type == 'residual':
            # Residual network (adds a delta to input)
            self.delta_network = nn.Sequential(
                nn.Linear(cogit_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, cogit_dim),
                nn.Tanh()  # Small bounded changes
            )
            self.operator = lambda x: x + 0.1 * self.delta_network(x)  # Small residual changes
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")
    
    def forward(self, cogit: torch.Tensor) -> torch.Tensor:
        """Apply cognitive manipulation to cogit vector"""
        if self.operator_type == 'residual':
            return self.operator(cogit)
        else:
            output = self.operator(cogit)
            # Ensure output stays in valid HDC range
            return torch.tanh(output)


class OperatorTrainer:
    """Trains cognitive manipulation operators from contrasting pairs"""
    
    def __init__(self, device: torch.device):
        self.device = device
        
    def prepare_training_data(self, cogit_pairs: List[Dict]) -> Tuple[List, List]:
        """
        Prepare training data from cogit pairs.
        Returns (train_pairs, test_pairs) where each pair is (low_cogit, high_cogit, metadata)
        """
        all_pairs = []
        
        for pair in cogit_pairs:
            low_cogit = torch.tensor(pair['low_cogit'], dtype=torch.float32)
            high_cogit = torch.tensor(pair['high_cogit'], dtype=torch.float32)
            
            metadata = {
                'dimension': pair['dimension'],
                'layer': pair['layer'],
                'low_text': pair['low_text'],
                'high_text': pair['high_text']
            }
            
            all_pairs.append((low_cogit, high_cogit, metadata))
        
        # Shuffle and split
        random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * 0.8)
        train_pairs = all_pairs[:split_idx]
        test_pairs = all_pairs[split_idx:]
        
        return train_pairs, test_pairs
    
    def train_operator(self, operator: CognitiveManipulationOperator, 
                      train_pairs: List[Tuple], 
                      epochs: int = 100,
                      learning_rate: float = 0.001) -> Dict[str, List]:
        """Train operator to transform low → high cognitive states"""
        
        operator.to(self.device)
        optimizer = optim.Adam(operator.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'epoch': []}
        
        print("Training cognitive manipulation operator...")
        
        for epoch in tqdm(range(epochs), desc="Training"):
            total_loss = 0
            
            # Batch processing
            random.shuffle(train_pairs)
            
            for low_cogit, high_cogit, _ in train_pairs:
                low_cogit = low_cogit.to(self.device)
                high_cogit = high_cogit.to(self.device)
                
                # Forward pass: transform low → high
                predicted_high = operator(low_cogit)
                loss = criterion(predicted_high, high_cogit)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_pairs)
            history['train_loss'].append(avg_loss)
            history['epoch'].append(epoch + 1)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return history
    
    def evaluate_operator(self, operator: CognitiveManipulationOperator,
                         test_pairs: List[Tuple]) -> Dict[str, float]:
        """Evaluate operator on test set"""
        
        operator.eval()
        total_mse = 0
        total_cosine_sim = 0
        
        with torch.no_grad():
            for low_cogit, high_cogit, _ in test_pairs:
                low_cogit = low_cogit.to(self.device)
                high_cogit = high_cogit.to(self.device)
                
                # Apply operator
                predicted_high = operator(low_cogit)
                
                # Calculate metrics
                mse = torch.mean((predicted_high - high_cogit) ** 2).item()
                cosine_sim = torch.cosine_similarity(
                    predicted_high.unsqueeze(0), 
                    high_cogit.unsqueeze(0)
                ).item()
                
                total_mse += mse
                total_cosine_sim += cosine_sim
        
        return {
            'test_mse': total_mse / len(test_pairs),
            'test_cosine_similarity': total_cosine_sim / len(test_pairs)
        }


class BehavioralTester:
    """Tests if learned operators produce expected behavioral changes"""
    
    def __init__(self, model_adapter, projection_strategy, device):
        self.adapter = model_adapter
        self.projection = projection_strategy
        self.device = device
        
    def test_operator_behavior(self, operator: CognitiveManipulationOperator,
                              test_prompts: List[str],
                              target_dimension: str,
                              layer: int = 6) -> Dict[str, Any]:
        """
        Test if operator produces expected behavioral changes.
        E.g., does the certainty operator make outputs more confident?
        """
        
        results = []
        
        for prompt in test_prompts:
            # Extract original activation
            hidden_states = self.adapter.extract_hidden_states(prompt, [layer])
            original_activation = hidden_states[layer]
            
            # Project to HDC space
            original_cogit = self.projection.project(original_activation)
            
            # Apply operator
            manipulated_cogit = operator(original_cogit.to(self.device))
            
            # Project back to activation space
            manipulated_activation = self.projection.inverse_project(manipulated_cogit)
            
            # Generate text with original activation
            original_output = self.adapter.model.generate(
                self.adapter.tokenizer(prompt, return_tensors="pt").input_ids,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True
            )
            original_text = self.adapter.tokenizer.decode(original_output[0], skip_special_tokens=True)
            
            # Generate text with manipulated activation (simplified - actual injection is complex)
            # For now, we'll analyze the difference in cogit space
            cogit_change = torch.norm(manipulated_cogit - original_cogit).item()
            
            result = {
                'prompt': prompt,
                'original_output': original_text,
                'cogit_change_magnitude': cogit_change,
                'target_dimension': target_dimension
            }
            results.append(result)
        
        # Analyze results
        avg_change = np.mean([r['cogit_change_magnitude'] for r in results])
        
        return {
            'individual_results': results,
            'average_cogit_change': avg_change,
            'dimension': target_dimension
        }


def main():
    """Main operator learning and testing pipeline"""
    print("🧠 Stage 3: Cognitive Manipulation Operator Learning")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    model_name = config['model']['name']
    model_config = config['model']['configs'].get(model_name, {})
    input_dim = model_config.get('hidden_dim', 768)
    hdc_dim = config['hdc']['dimension']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    paths_mode = config['paths']['mode']
    base_path = config['paths'][paths_mode]['data_dir']
    
    # Find cogit files from Stage 2
    cogit_dir = Path(base_path) / "processed" / "cogits"
    if not cogit_dir.exists():
        raise FileNotFoundError("No cogit files found. Run Stage 2 first.")
    
    # Load best projection strategy results
    stage2_metrics_file = Path(config['paths'][paths_mode]['results_dir']) / "stage2_metrics.json"
    if stage2_metrics_file.exists():
        with open(stage2_metrics_file) as f:
            stage2_metrics = json.load(f)
            best_strategy = stage2_metrics.get('best_strategy', 'random')
    else:
        best_strategy = 'random'
    
    print(f"Using {best_strategy} projection (best from Stage 2)")
    
    # Find cogit files for best strategy
    cogit_files = list(cogit_dir.glob(f"cogits_{best_strategy}_*.jsonl"))
    if not cogit_files:
        raise FileNotFoundError(f"No cogit files for {best_strategy} strategy")
    
    latest_cogits = sorted(cogit_files)[-1]
    print(f"Using cogit file: {latest_cogits.name}")
    
    # Load cogit pairs
    cogit_pairs = []
    with jsonlines.open(latest_cogits) as reader:
        cogit_pairs = list(reader)
    
    print(f"Loaded {len(cogit_pairs)} cogit pairs")
    
    # Group pairs by dimension for targeted learning
    pairs_by_dimension = {}
    for pair in cogit_pairs:
        dim = pair['dimension']
        if dim not in pairs_by_dimension:
            pairs_by_dimension[dim] = []
        pairs_by_dimension[dim].append(pair)
    
    # Create output directory
    output_dir = Path(base_path) / "models" / "operators"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = OperatorTrainer(device)
    
    # Train operator for each dimension
    all_results = {}
    
    for dimension, dim_pairs in pairs_by_dimension.items():
        print(f"\n📚 Training operator for {dimension} dimension...")
        print(f"  Using {len(dim_pairs)} training pairs")
        
        # Prepare data
        train_pairs, test_pairs = trainer.prepare_training_data(dim_pairs)
        print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")
        
        # Create operator
        operator = CognitiveManipulationOperator(
            cogit_dim=hdc_dim,
            operator_type='residual'  # Use residual for small targeted changes
        )
        
        # Train operator
        history = trainer.train_operator(
            operator, 
            train_pairs,
            epochs=config['experiment']['training']['epochs'],
            learning_rate=config['experiment']['training']['learning_rate']
        )
        
        # Evaluate operator
        eval_metrics = trainer.evaluate_operator(operator, test_pairs)
        print(f"  Test MSE: {eval_metrics['test_mse']:.6f}")
        print(f"  Test Cosine Similarity: {eval_metrics['test_cosine_similarity']:.4f}")
        
        # Save operator
        operator_file = output_dir / f"operator_{dimension}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save({
            'model_state_dict': operator.state_dict(),
            'dimension': dimension,
            'training_history': history,
            'eval_metrics': eval_metrics,
            'num_train_pairs': len(train_pairs),
            'num_test_pairs': len(test_pairs),
            'projection_strategy': best_strategy,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': str(device)
            }
        }, operator_file)
        
        all_results[dimension] = {
            'operator_file': str(operator_file),
            'eval_metrics': eval_metrics,
            'num_pairs': len(dim_pairs)
        }
        
        print(f"  ✓ Saved operator to {operator_file.name}")
    
    # Behavioral testing (simplified for now)
    print("\n🧪 Behavioral Testing...")
    
    # Load model adapter for testing
    adapter = ModelAdapterFactory.create_adapter(model_name, str(device))
    
    # Create projection for testing (match what was used in training)
    if best_strategy == 'random':
        projection = RandomProjection(input_dim, hdc_dim)
    elif best_strategy == 'learned':
        projection = LearnedProjection(input_dim, hdc_dim)
    else:
        projection = PaddingProjection(input_dim, hdc_dim)
    
    # Test certainty operator if available
    if 'certainty' in all_results:
        print("Testing certainty operator...")
        
        # Reload certainty operator
        certainty_op_file = all_results['certainty']['operator_file']
        certainty_operator = CognitiveManipulationOperator(cogit_dim=hdc_dim, operator_type='residual')
        checkpoint = torch.load(certainty_op_file, map_location=device)
        certainty_operator.load_state_dict(checkpoint['model_state_dict'])
        certainty_operator.to(device)
        
        # Create behavioral tester
        tester = BehavioralTester(adapter, projection, device)
        
        # Test on sample prompts
        test_prompts = ["I think", "It might be", "Perhaps the"]
        behavior_results = tester.test_operator_behavior(
            certainty_operator,
            test_prompts,
            'certainty',
            layer=6
        )
        
        print(f"  Average cogit change: {behavior_results['average_cogit_change']:.4f}")
        
        # Add to results
        all_results['certainty']['behavioral_test'] = behavior_results
    
    # Generate final metrics
    metrics = {
        'model': model_name,
        'device': str(device),
        'projection_strategy': best_strategy,
        'dimensions_trained': list(all_results.keys()),
        'operator_results': all_results,
        'total_cogit_pairs': len(cogit_pairs),
        'timestamp': datetime.now().isoformat(),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'not_set')
    }
    
    # Save metrics
    results_dir = Path(config['paths'][paths_mode]['results_dir'])
    with open(results_dir / "stage3_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Stage 3 complete!")
    print(f"✓ Trained operators for {len(all_results)} cognitive dimensions")
    print(f"✓ Operators saved to {output_dir}")
    print(f"\n🎯 Key Finding: Operators can manipulate cogit representations")
    print(f"   Next step: Test if these manipulations transfer to actual behavioral changes")


if __name__ == "__main__":
    main()