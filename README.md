# Cognitive Manipulation via Learned HDC Operators

## Study Design and Findings

Understanding how to mathematically manipulate cognitive states in artificial neural networks has direct applications to brain-computer interfaces, where similar operators might modify biological neural activity to achieve targeted cognitive changes. This research investigates the fundamental question of whether learned mathematical transformations can predictably alter cognition while preserving structural coherence.

We tested whether we could learn mathematical operators to steer GPT-2 sentiment while keeping text readable. We extracted GPT-2's internal representations (768-dimensional vectors) from sentiment prompts and converted them into 10,000-dimensional binary vectors (+1/-1) using random projection - this high-dimensional format preserves meaning while making the vectors easier to manipulate mathematically. We then trained neural networks to transform positive sentiment vectors toward negative ones in this high-dimensional space. We tested two datasets: 10 simple prompts per sentiment ("I love my puppy" vs "That was terrible") and 50 diverse prompts with varied structures and contexts. For intervention, we intercepted GPT-2's activations during generation, converted them to high-dimensional space, applied our learned operator, converted back to the original format, and blended the results in.

The 10-prompt dataset failed - the operator overfitted and produced gibberish (dashes and punctuation instead of words). The 50-prompt dataset worked. When we applied the trained operator to neutral prompts like "The restaurant downtown is," baseline outputs were positive ("delicious local craft fare") but interventions introduced criticism ("mixed bag... there are a few things I would change"). The steering was controllable through blend ratios. Data diversity, not model complexity, determined whether we could manipulate sentiment while preserving coherence.

## ⚠️ Codebase Warning: Very Messy

This repository contains a mix of working experimental code and broken/unused files from development. Most directories and files should be ignored.

### ✅ WORKING FILES - Look at these:

**Core Architecture:**
- `src/model_adapter_tl.py` - TransformerLens implementation that actually works

**Sentiment Manipulation Experiment (the actual research):**
- `generate_diverse_prompts.py` - Creates the 50 balanced training prompts
- `sentiment_experiment.py` - Baseline 10-example experiment (demonstrates overfitting)
- `sentiment_experiment_improved.py` - Phase 1: Collects diverse data from 50 prompts  
- `fix_balance.py` - Fixes data imbalance (50 positive, 55 negative → 50/50)
- `sentiment_phase2_improved.py` - Phase 2: Trains the mathematical operator
- `sentiment_phase3_improved.py` - Phase 3: Tests intervention with blending

**Data and Results:**
- `data/sentiment_experiment/diverse_prompts_50.json` - The 50 training prompts
- `data/sentiment_experiment/balanced_cogits_20250903_035408.json` - Final training data
- `models/sentiment_operator/robust_sentiment_operator.pt` - Trained operator (64MB)
- `results/sentiment_intervention/improved_results.json` - Final experimental results

### ❌ IGNORE THESE FILES - Broken/unused development artifacts:

**Broken Infrastructure:**
- `src/model_adapter.py` - Manual hook implementation (caused infinite hanging)
- `src/stage1_simulation/` - Original unused framework
- `src/stage2_encoding/` - Original unused framework  
- `src/stage3_learning/` - Original unused framework

**Failed Development Attempts:**
- `test_*.py`, `debug_*.py`, `minimal_test.py` - Hook debugging attempts
- `working_injection.py`, `fix_injection.py` - Manual hook failures
- `sentiment_phase2_efficient.py` - Earlier training version
- `sentiment_phase3_intervention.py` - Earlier intervention version
- `compare_implementations.py` - Development comparison

**Empty/Unused Directories:**
- `embeddings/` - Empty
- `notebooks/` - Empty
- `scripts/` - Empty
- `docs/` - Empty  
- `infra/` - Empty
- `models/operators/` - Original framework operators (unused)
- `data/raw/`, `data/processed/`, `data/models/` - Original pipeline data (unused)

## How to Run the Working Experiment

```bash
# 1. Generate training prompts (if needed)
python generate_diverse_prompts.py

# 2. Phase 1: Collect activation data  
python sentiment_experiment_improved.py

# 3. Fix data balance (if needed)
python fix_balance.py

# 4. Phase 2: Train the operator
python sentiment_phase2_improved.py

# 5. Phase 3: Test intervention
python sentiment_phase3_improved.py
```

## For HDC Verification - Exact Code Locations

**HDC Encoding Process** (verify the math here):
- `sentiment_experiment_improved.py:118-146` - HDC encoding function with random projection
- `sentiment_experiment_improved.py:93-97` - Projection matrix creation (deterministic, seed=42)
- `sentiment_experiment_improved.py:107-110` - Sign binarization: `cogit = torch.sign(hd_vector)`

**HDC Decoding Process**:
- `sentiment_phase3_improved.py:194-206` - HDCDecoder class with pseudoinverse
- `sentiment_phase3_improved.py:201-205` - Decode function: `activation = torch.matmul(cogit, self.inverse_projection)`

**Critical HDC Operations**:
- **Encoding**: 768-dim activation → multiply by random projection matrix → sign() → 10K-dim binary cogit
- **Transform**: Neural network operates directly on 10K-dim cogit space  
- **Decoding**: cogit → multiply by pseudoinverse of projection → 768-dim activation
- **Blending**: `(1-ratio) * original + ratio * modified` in original activation space

**Verify These Lines Specifically**:
- `sentiment_experiment_improved.py:108` - `hd_vector = torch.matmul(activation, self.projection)`
- `sentiment_experiment_improved.py:110` - `cogit = torch.sign(hd_vector)`  
- `sentiment_phase3_improved.py:204` - `activation = torch.matmul(cogit, self.inverse_projection)`

## Key Finding

Mathematical operators can learn to manipulate language model cognition, but only with diverse training data (50 examples worked, 10 failed). The success depends on data diversity, not architectural complexity.

## Research Team

**Research by**: Bryce-Allen Bagley & Austin Morrissey  
**Implementation assistance**: Claude Opus 4.1