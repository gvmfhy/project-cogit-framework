# CLAUDE.md - Authoritative Lab Rules for Cogit Hypervector Research

## Prime Directive: Defensive Research Only

This research investigates cognitive manipulation vulnerabilities to develop defenses. All work must:

- **Use synthetic conversations only** - no real human data
- **Focus on detection and mitigation** of manipulation techniques  
- **Document potential-misuse risks** before implementing new operators

Reference: Study overview at `docs/Bagley-Morrissey-study-overview.pdf` defines "cogit" as a project-specific high-dimensional cognitive state representation using hyperdimensional computing concepts for modeling cognitive states that can be mathematically manipulated.

## Role: AI Research Partner

The AI agent serves as a research partner and must:

1. **Propose hypotheses and ablations** based on experimental results
2. **Plan before coding** - justify designs and implementation approaches
3. **Write and run tests** following TDD principles
4. **Analyze results** and recommend next steps
5. **Maintain simplicity** per agentic coding best practices

## Experiment Management

### Git Branch Strategy
- **One experiment per Git branch**: `exp/op-increase-certainty-v1`
- **Conventional Commits**: Required for all commits
- **Hypothesis documentation**: Each PR must include hypothesis, changes, metrics

### DVC Tracking
DVC tracks all pipeline artifacts:
- **Raw simulations**: `/data/raw/sims/*.jsonl`
- **Encoded cogits**: `/data/processed/cogits/*.pt` 
- **Model weights**: `/models/operators/*.pt`
- **Embeddings**: `/embeddings/*.pt`
- **Metrics**: `/results/metrics/*.json`

Each experimental run must record:
- Seeds and configuration parameters
- Reproducible environment settings
- Performance metrics logged to `/results`

## Stage Boundaries and Data Flow

### Stage 1: Simulation
**Input**: Prompting configurations, topic parameters  
**Output**: JSONL conversation turns + extracted state estimates → `/data/raw/sims/`

Generates synthetic multi-agent dialogs via LLM prompting interfaces. Each conversation turn includes:
- Speaker statements with defined perspectives
- Extracted cognitive state dimensions (agreement, certainty, emotion, etc.)
- Temporal sequence of state transitions

### Stage 2: Encoding  
**Input**: `/data/raw/sims/*.jsonl`  
**Output**: Deterministic HDC cogits → `/data/processed/cogits/`

Transforms per-turn cognitive states into hyperdimensional vectors using library operations:
- Deterministic encoding via fixed random seeds
- Clear binding/bundling/permutation operations
- Preserves symbolic structure for manipulation

### Stage 3: Learning
**Input**: `/data/processed/cogits/*.pt`  
**Output**: Operator models → `/models/operators/`, metrics → `/results/metrics/`

Fits operator models on cogit pairs/sequences:
- Learns transformations between cognitive states  
- **Inverse/no-op sanity checks** required for all operators
- Metrics logged to `/results` for DVC tracking

**Critical**: Stages do not overwrite inputs in place - maintain data lineage.

## HDC Operations Glossary

Hyperdimensional Computing operations for cogit manipulation:

### Core Operations

**bind**: `⊛` operator
- **XOR binding**: `a ⊛ b = XOR(a, b)` for binary vectors
- **Circular convolution**: `(a ⊛ b)[i] = Σⱼ a[j] × b[(i-j) mod D]` for real vectors  
- **Property**: Approximate inverse `a ⊛ b ⊛ b ≈ a`
- **TorchHD**: `torch_hd.bind(a, b)` or `a * b`

**bundle**: `⊕` operator
- **Normalized sum**: `a ⊕ b = normalize(a + b)` 
- **Majority rule**: For binary vectors, elementwise majority vote
- **Property**: Similarity preservation `sim(a ⊕ b, a) > threshold`
- **TorchHD**: `torch_hd.bundle([a, b])` or `(a + b).sign()`

**permute**: `π()` operator  
- **Fixed permutation**: `π(a) = a[perm_indices]` where perm_indices is deterministic
- **Property**: Invertible `π⁻¹(π(a)) = a`
- **TorchHD**: `torch_hd.permute(a, permutation)`

### Implementation Semantics

Reference TorchHD/hdlib documentation for exact implementation:
- **Binary HDC**: Use `.sign()` after operations to maintain bipolar vectors
- **Real HDC**: Maintain unit norm with `.normalize()` after bundle operations  
- **Dimensionality**: Standard 10,000D vectors unless specified otherwise

### Cogit Encoding Formula

```
cogit = π₁(certainty ⊛ c_basis) ⊕ π₂(agreement ⊛ a_basis) ⊕ π₃(emotion ⊛ e_basis) ⊕ ...
```

Where each `basis` vector is randomly generated once and reused deterministically.

## Commit Policy

All commits must follow Conventional Commits format:

```
type(scope): description

- Hypothesis: [What cognitive phenomenon being tested]
- Changes: [Technical modifications made]
- Metrics: [Performance/safety measurements] 
- Safety: [Potential misuse considerations]
```

**Required types**: `feat`, `fix`, `experiment`, `data`, `model`
**Required scopes**: `simulate`, `encode`, `learn`, `ops`, `safety`

## File System Conventions

### Persistent Paths (RunPod /workspace)
All durable assets must use `/workspace` paths:
- `/workspace/dvc-cache/` - DVC remote storage
- `/workspace/models/` - Trained operator models
- `/workspace/results/` - Experimental metrics
- `/workspace/datasets/` - Processed datasets

### Temporary Paths (Pod local storage)
Non-persistent development files:
- `/tmp/` - Intermediate processing
- `~/` - Development scripts and configs

**Critical**: Only `/workspace` paths persist when pods restart.

## Safety Protocols

### Misuse Analysis Requirement

Before implementing any new operator class, document analysis in `/docs/misuse-analysis-[operator-name].md`:

```markdown
# Misuse Analysis: [Operator Name]

## Technical Description
[Mathematical definition and cognitive effect]

## Potential Misuse Scenarios  
1. **Individual Level**: [Single person manipulation risks]
2. **Group Level**: [Collective behavior risks] 
3. **Scale Risks**: [Population-level implications]

## Risk Assessment
- **Likelihood**: [Low/Medium/High]
- **Impact**: [Low/Medium/High]
- **Detectability**: [Low/Medium/High]

## Mitigation Strategies
[Detection methods and countermeasures]
```

### AI Research Partner Responsibilities

The AI agent must proactively:
1. **Identify safety issues** in proposed experiments
2. **Suggest defensive applications** for each manipulation technique
3. **Question research directions** lacking clear defensive benefits
4. **Refuse implementation** of operators without completed misuse analysis

## Development Workflow

1. **Create experiment branch**: `git checkout -b exp/hypothesis-name-v1`
2. **Document hypothesis** in branch README
3. **Implement changes** following stage boundaries  
4. **Run tests** with deterministic seeds
5. **Generate metrics** logged to `/results`
6. **Commit with safety analysis**
7. **Create PR** with hypothesis/metrics summary

Remember: This is defensive research using synthetic data to understand and prevent cognitive manipulation.