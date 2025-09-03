# Cognitive Manipulation via Learned HDC Operators

## Study Design and Findings

Understanding how to mathematically manipulate cognitive states in artificial neural networks has direct applications to brain-computer interfaces, where similar operators might modify biological neural activity to achieve targeted cognitive changes. We tested whether we could learn mathematical operators to steer GPT-2 sentiment while keeping text readable.

Data Generation Phase
We created paired positive and negative prompts to extract contrasting cognitive states:
  Positive prompts: "I love my new puppy, he is so...", "That was a wonderful and happy..."
  Negative prompts: "The traffic was horrible this morning, it was...", "I had a terrible and awful..."

We fed these prompts through GPT-2 and used TransformerLens to capture the model's internal activations from Layer 6. We'll convert all of these captured activations into a hyperdimensional computing space (HDC), and refer to them as cogit vectors. These vectors, in the sense of their sentiment, should be directionally opposed. 

Next, we use these vectors to determine  mathematical "dial" that can turn a positive thought into a negative one.
1.	Train a Transformation: We will train a simple machine learning model. Its only job is to learn the mathematical recipe (the "operator") that transforms a positive cogit vector into its corresponding negative cogit vector.
2.	The Result: At the end of this phase, you will have a single, learned mathematical function: make_negative_operator. This function now represents the abstract concept of "making something more negative."
3.	We then test this operator on sentences the model was not trained on, namely, neutral unfinished thoughts.

When training data is small, the model overfits, and intervention renders incoherent response. When training data is increased, say, from 5 pairs of prompts to 50 pairs of prompts, coherence and directionality is maintained. Thus, this gives us indicator of a proof of concept that small, subtle interventions are an attack vector for cognitive securtiy. We expect to replicate these findings robustly in stronger models. 

## ⚠️ Codebase Warning: Very Messy Pilot

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

Math can steer models. It is likely to steer minds. 

## Research Team

**Research by**: Bryce-Allen Bagley & Austin Morrissey  
**Implementation assistance**: Claude Opus 4.1
