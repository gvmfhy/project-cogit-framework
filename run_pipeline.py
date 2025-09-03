#!/usr/bin/env python3
"""Run the complete 3-stage pipeline"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONHASHSEED'] = '42'
os.environ['PYTHONWARNINGS'] = 'ignore'

import subprocess
import sys

def run_stage(stage_num, script_path):
    """Run a single stage and report results"""
    print(f"\n{'='*60}")
    print(f"STAGE {stage_num}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Show last 20 lines of output
        lines = result.stdout.strip().split('\n')
        for line in lines[-20:]:
            print(line)
        print(f"\n✓ Stage {stage_num} completed successfully")
        return True
    else:
        print(f"✗ Stage {stage_num} failed")
        print("Error output:")
        print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        return False

def main():
    print("🚀 Running Complete Cogit Pipeline")
    print("="*60)
    
    stages = [
        (1, "src/stage1_simulation/run.py"),
        (2, "src/stage2_encoding/run.py"),
        (3, "src/stage3_learning/run.py")
    ]
    
    for stage_num, script_path in stages:
        success = run_stage(stage_num, script_path)
        if not success:
            print(f"\n❌ Pipeline stopped at Stage {stage_num}")
            return 1
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print("\nResults saved in:")
    print("  - data/raw/activations/     (Stage 1)")
    print("  - data/processed/cogits/    (Stage 2)")
    print("  - models/operators/         (Stage 3)")
    print("  - results/                  (Metrics)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())