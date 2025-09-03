#!/usr/bin/env python3
"""
Stage 1: Simulation - Generate synthetic multi-agent dialogs and per-turn state estimates
Produces JSONL artifacts for encoding stage.
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
from typing import Dict, List, Any
from transformers import pipeline, set_seed
import yaml

# Deterministic seeding
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
set_seed(42)

def load_params() -> Dict[str, Any]:
    """Load simulation parameters from params.yaml"""
    params_file = Path("params.yaml")
    if params_file.exists():
        with open(params_file) as f:
            params = yaml.safe_load(f)
            return params.get('simulate', {})
    
    # Default parameters
    return {
        'num_conversations': 10,
        'turns_per_conversation': 8,
        'topics': [
            'climate change policy',
            'artificial intelligence regulation', 
            'social media privacy',
            'education funding'
        ]
    }

def extract_cognitive_state(text: str, speaker_id: str, turn: int) -> Dict[str, float]:
    """
    Extract cognitive state dimensions from speaker text.
    In a full implementation, this would use LLM prompting or activation analysis.
    """
    # Deterministic state extraction based on text features
    text_hash = hash(f"{text}{speaker_id}{turn}") % 1000
    
    return {
        'agreement': (text_hash % 100) / 100.0 * 2 - 1,  # -1 to 1
        'certainty': abs(text_hash % 50) / 50.0,  # 0 to 1  
        'openness': (text_hash % 80) / 80.0,  # 0 to 1
        'emotional_tone': ((text_hash % 60) - 30) / 30.0,  # -1 to 1
        'social_alignment': ((text_hash % 40) - 20) / 20.0,  # -1 to 1
    }

def simulate_conversation(topic: str, conversation_id: int, turns: int) -> List[Dict[str, Any]]:
    """Simulate a multi-agent conversation on a topic"""
    speakers = ['Agent_A', 'Agent_B', 'Agent_C']
    conversation = []
    
    for turn in range(turns):
        speaker = speakers[turn % len(speakers)]
        
        # Generate synthetic statement (placeholder - would use LLM in full implementation)
        statement = f"Turn {turn+1} statement by {speaker} on {topic}. " \
                   f"This represents a synthetic dialog turn for cognitive state modeling."
        
        # Extract cognitive state for this turn
        cogit_state = extract_cognitive_state(statement, speaker, turn)
        
        turn_data = {
            'conversation_id': conversation_id,
            'turn': turn + 1,
            'speaker': speaker,
            'statement': statement,
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'cognitive_state': cogit_state
        }
        
        conversation.append(turn_data)
    
    return conversation

def main():
    """Main simulation pipeline"""
    print("🎭 Stage 1: Starting Simulation Pipeline")
    print("=" * 50)
    
    # Load parameters
    params = load_params()
    print(f"Parameters: {params}")
    
    # Create output directory
    output_dir = Path("data/raw/sims")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate conversations
    all_conversations = []
    total_turns = 0
    
    for conv_id in range(params['num_conversations']):
        topic = random.choice(params['topics'])
        conversation = simulate_conversation(
            topic, 
            conv_id, 
            params['turns_per_conversation']
        )
        all_conversations.extend(conversation)
        total_turns += len(conversation)
        print(f"Generated conversation {conv_id + 1}/{params['num_conversations']}: {topic}")
    
    # Save to JSONL
    output_file = output_dir / f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(all_conversations)
    
    # Generate metrics
    metrics = {
        'total_conversations': params['num_conversations'],
        'total_turns': total_turns,
        'unique_topics': len(set(params['topics'])),
        'avg_turns_per_conversation': total_turns / params['num_conversations'],
        'output_file': str(output_file),
        'timestamp': datetime.now().isoformat(),
        'seed': 42,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', 'not_set')
    }
    
    # Save metrics
    metrics_dir = Path("results")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "simulate_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Simulation complete: {total_turns} turns saved to {output_file}")
    print(f"✓ Metrics saved to results/simulate_metrics.json")
    print(f"✓ PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")

if __name__ == "__main__":
    main()