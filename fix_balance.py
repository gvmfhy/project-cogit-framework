#!/usr/bin/env python3
"""
Fix the 50 vs 55 imbalance issue
"""

import json
from pathlib import Path

def fix_imbalance():
    """Balance the dataset to exactly 50 positive and 50 negative"""
    
    print("🔧 Fixing dataset imbalance...")
    
    # Load the latest improved dataset
    data_dir = Path("data/sentiment_experiment")
    cogit_files = list(data_dir.glob("improved_cogits_*.json"))
    latest_file = max(cogit_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Check current sizes
    pos_count = len(data['positive_cogits'])
    neg_count = len(data['negative_cogits'])
    
    print(f"Current: {pos_count} positive, {neg_count} negative")
    
    # Balance by trimming to minimum
    min_count = min(pos_count, neg_count)
    
    balanced_data = {
        'positive_cogits': data['positive_cogits'][:min_count],
        'negative_cogits': data['negative_cogits'][:min_count],
        'hd_dim': data['hd_dim'],
        'dataset_size': f"{min_count} positive + {min_count} negative",
        'improvement': 'diverse_prompts_balanced_50_examples',
        'timestamp': data['timestamp'] + '_balanced'
    }
    
    # Save balanced dataset
    output_file = data_dir / f"balanced_cogits_{data['timestamp']}.json"
    with open(output_file, 'w') as f:
        json.dump(balanced_data, f)
    
    print(f"✅ Balanced dataset: {min_count} positive, {min_count} negative")
    print(f"💾 Saved to: {output_file.name}")
    
    return output_file

if __name__ == "__main__":
    fix_imbalance()