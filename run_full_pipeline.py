#!/usr/bin/env python3
"""
Full Pipeline Runner with TransformerLens
Runs all three stages in sequence with the new clean implementation.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_stage(stage_num: int, script_name: str, description: str):
    """Run a pipeline stage"""
    print("\n" + "=" * 70)
    print(f"STAGE {stage_num}: {description}")
    print("=" * 70)
    
    # Run the stage script
    result = subprocess.run(
        ["python", script_name],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()}
    )
    
    if result.returncode != 0:
        print(f"❌ Stage {stage_num} failed!")
        print("Error output:")
        print(result.stderr)
        return False
    
    # Show key outputs
    output_lines = result.stdout.split('\n')
    for line in output_lines:
        if '✓' in line or 'Complete' in line:
            print(f"  {line.strip()}")
    
    return True

def main():
    print("=" * 70)
    print("COGNITIVE MANIPULATION FRAMEWORK - FULL PIPELINE")
    print("Powered by TransformerLens (No More Deadlocks!)")
    print("=" * 70)
    
    # Ensure we're in the venv_clean environment
    if not os.path.exists("venv_clean"):
        print("❌ Clean environment not found! Run setup first.")
        return
    
    # Stage 1: Activation Extraction
    if not run_stage(
        1, 
        "src/stage1_simulation/run_tl.py",
        "ACTIVATION EXTRACTION (TransformerLens)"
    ):
        return
    
    # Stage 2: HDC Encoding
    if not run_stage(
        2,
        "src/stage2_encoding/run_tl.py",
        "HYPERDIMENSIONAL ENCODING"
    ):
        return
    
    # Stage 3: Operator Learning
    if not run_stage(
        3,
        "src/stage3_learning/run_tl.py",
        "COGNITIVE OPERATOR LEARNING"
    ):
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE - ALL STAGES SUCCESSFUL!")
    print("=" * 70)
    
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 40)
    
    # Check generated files
    activations = list(Path("data/raw/sims").glob("activations_*.jsonl"))
    cogits = list(Path("data/processed/cogits").glob("cogits_*.jsonl"))
    operators = list(Path("models/operators").glob("operator_*.pt"))
    
    if activations:
        latest_act = max(activations, key=lambda p: p.stat().st_mtime)
        print(f"✓ Activations: {latest_act.name}")
    
    if cogits:
        latest_cog = max(cogits, key=lambda p: p.stat().st_mtime)
        print(f"✓ Cogits: {latest_cog.name}")
    
    if operators:
        print(f"✓ Operators: {len(operators)} trained models")
        for op in operators:
            print(f"  - {op.name}")
    
    print("\n🚀 KEY IMPROVEMENTS:")
    print("-" * 40)
    print("1. No hook deadlocks - TransformerLens handles everything")
    print("2. Clean environment - Only 15 essential packages")
    print("3. Reproducible - Deterministic seeds throughout")
    print("4. Stable - No hanging on generation with interventions")
    
    print("\n💡 NEXT STEPS:")
    print("-" * 40)
    print("1. Test cognitive manipulation with real prompts")
    print("2. Analyze operator effectiveness")
    print("3. Implement safety checks for manipulation detection")
    print("4. Document findings for defensive applications")
    
    print("\n" + "=" * 70)
    print("The framework is ready for cognitive manipulation research!")
    print("Remember: This is defensive research to understand and prevent manipulation.")
    print("=" * 70)

if __name__ == "__main__":
    # Activate the clean environment
    activate_cmd = "source venv_clean/bin/activate"
    print(f"Note: Make sure you're using the clean environment:")
    print(f"  {activate_cmd}")
    print()
    
    main()