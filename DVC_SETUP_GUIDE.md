# DVC Setup and Data Versioning Strategy for Cognitive Manipulation Research

## Overview

This guide provides comprehensive setup instructions and data versioning strategies for the cognitive manipulation research tool using DVC (Data Version Control). The pipeline is designed for reproducible research with support for multiple models, projection strategies, and experimental configurations.

## 🚀 Quick Setup

### 1. Initialize DVC (if not already done)

```bash
# Initialize DVC in your project
cd project-cogit-framework
dvc init

# Add remote storage (choose one based on your setup)
# Local storage
dvc remote add -d local /path/to/local/storage

# S3 storage (recommended for collaboration)
dvc remote add -d s3remote s3://your-bucket/cogit-research/

# Google Drive (for small teams)
dvc remote add -d gdrive gdrive://your-folder-id

# Configure remote
dvc remote modify s3remote region us-west-2  # if using S3
```

### 2. Install Additional Requirements

```bash
# Install DVC with specific extras based on your remote
pip install dvc[s3]  # for S3
pip install dvc[gs]  # for Google Cloud
pip install dvc[gdrive]  # for Google Drive

# Install additional dependencies for advanced features
pip install dvc[plots]  # for DVC plots
pip install dvclive  # for live metrics tracking
```

## 📊 Data Versioning Strategy

### Core Principles

1. **Immutable Datasets**: Raw activations and processed cogits are versioned and immutable
2. **Staged Processing**: Each pipeline stage has clear input/output versioning
3. **Experiment Tracking**: All experiments are tagged and timestamped
4. **Model Agnostic**: Versioning works across different transformer models
5. **Reproducible Research**: Full reproducibility through deterministic seeding

### Directory Structure and Versioning

```
data/
├── raw/
│   └── activations/          # DVC tracked, versioned by experiment
│       ├── exp_20250903_v1/  # Experiment-specific directories
│       └── exp_20250903_v2/
├── processed/
│   └── cogits/               # DVC tracked, strategy-specific versioning
│       ├── random_v1/
│       ├── learned_v1/
│       └── pca_v1/
└── external/                 # External datasets (DVC tracked separately)
    └── validation_sets/

models/
├── cache/                    # Model weights (DVC tracked with special handling)
└── operators/                # Trained operators (DVC tracked, timestamped)

results/
├── metrics/                  # DVC metrics (not cached, for comparison)
├── plots/                    # DVC plots (not cached, regenerated)
├── validation/               # Validation results (not cached for freshness)
└── analysis/                 # Comparative analysis (not cached)
```

## 🔄 Pipeline Execution Strategies

### Development Mode (Single Model, Single Strategy)

```bash
# Run with current config.yaml settings
dvc repro

# Run specific stages
dvc repro activation_extraction
dvc repro hdc_projection
dvc repro operator_learning
```

### Research Mode (Multiple Configurations)

```bash
# Experiment with different models
dvc repro -P model.name=gpt2
dvc repro -P model.name=llama2_7b

# Test different projection strategies
for strategy in random learned pca; do
    dvc repro -P projections.strategies=[$strategy]
done

# Comprehensive experiment (all combinations)
dvc exp run --queue -P model.name=gpt2 -P projections.strategies=[random]
dvc exp run --queue -P model.name=gpt2 -P projections.strategies=[learned]
dvc exp run --queue -P model.name=llama2_7b -P projections.strategies=[random]
dvc exp run --queue -P model.name=llama2_7b -P projections.strategies=[learned]

# Run all queued experiments
dvc exp run --run-all
```

### Reproducibility Mode

```bash
# Create experiment branch
git checkout -b experiment-certainty-manipulation

# Run experiment with full documentation
dvc repro --force  # Force re-run for clean results
dvc exp run --name "certainty_baseline_$(date +%Y%m%d)"

# Tag successful experiments
dvc exp apply exp-12345  # Apply specific experiment
git tag -a exp-certainty-v1.0 -m "Baseline certainty manipulation results"
```

## 🎯 Experiment Management

### Experiment Naming Convention

```
{dimension}_{model}_{projection}_{date}_{version}
Examples:
- certainty_gpt2_random_20250903_v1
- agreement_llama2_learned_20250903_v1
- emotion_mistral_pca_20250903_v2
```

### Tracking Experiments

```bash
# List all experiments
dvc exp show

# Compare experiments
dvc exp diff exp1 exp2

# Show experiment metrics
dvc exp show --include-metrics

# Export experiment data
dvc exp show --csv > experiments_comparison.csv
```

## 📈 Metrics and Monitoring

### Metrics Configuration

The pipeline automatically tracks:

- **Stage Metrics**: Performance metrics for each pipeline stage
- **Model Metrics**: Accuracy, loss, convergence metrics for operator learning
- **Behavioral Metrics**: Manipulation effectiveness, consistency scores
- **Resource Metrics**: Memory usage, GPU utilization, execution time

### Metrics Comparison

```bash
# Compare metrics across experiments
dvc metrics show
dvc metrics diff

# Generate plots
dvc plots show results/plots/
dvc plots diff experiment1 experiment2
```

## 🔒 Data Security and Compliance

### Sensitive Data Handling

1. **Model Weights**: Store in secure, access-controlled remote storage
2. **API Keys**: Use environment variables, never commit to repository
3. **Generated Text**: Review for sensitive content before committing

### Access Control

```bash
# Set up access control for S3 remote
dvc remote modify s3remote access_key_id YOUR_ACCESS_KEY
dvc remote modify s3remote secret_access_key YOUR_SECRET_KEY --local

# Use IAM roles for production
dvc remote modify s3remote use_ssl true
dvc remote modify s3remote sse AES256
```

## 🚀 Production Deployment

### Environment-Specific Configuration

```bash
# Development
dvc repro -P paths.mode=local

# RunPod/GPU Cloud
dvc repro -P paths.mode=runpod

# HPC Cluster
dvc repro -P paths.mode=cluster
```

### Automated Pipeline Execution

```bash
# Create automation script
cat > run_experiment.sh << 'EOF'
#!/bin/bash
set -e

# Setup environment
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0

# Run pipeline
dvc repro --force

# Archive results
DATE=$(date +%Y%m%d_%H%M%S)
dvc exp apply HEAD
git tag -a "auto-exp-${DATE}" -m "Automated experiment ${DATE}"

# Push to remote
dvc push
git push origin --tags
EOF

chmod +x run_experiment.sh
```

## 📝 Best Practices

### 1. Version Control Integration

```bash
# Always commit params and config changes before experiments
git add params.yaml config.yaml
git commit -m "Update experiment parameters for certainty manipulation"

# Run experiment
dvc repro

# Commit DVC changes
git add dvc.lock .dvc/cache
git commit -m "Complete certainty manipulation experiment"
```

### 2. Data Validation

```bash
# Validate data integrity
dvc data status
dvc status

# Check for data drift
python scripts/validate_data.py --input data/raw/activations/ --baseline data/baselines/
```

### 3. Resource Management

```bash
# Monitor resource usage
dvc repro --verbose  # Show resource usage
nvidia-smi  # Monitor GPU usage

# Clean up unnecessary cache
dvc cache dir  # Show cache location
dvc gc  # Clean unreferenced cache files
```

### 4. Collaboration Workflow

```bash
# Team member workflow
git pull
dvc pull  # Get latest data

# Make changes
# ... modify code, params ...

# Run experiment
dvc repro

# Share results
dvc push
git add .; git commit -m "Add new projection strategy results"
git push
```

## 🔧 Troubleshooting

### Common Issues

1. **Cache Location**: Ensure sufficient disk space for DVC cache
2. **Remote Access**: Verify remote storage credentials and permissions
3. **Memory Issues**: Adjust batch sizes in params.yaml for large models
4. **GPU Issues**: Verify CUDA availability and model device settings

### Debug Commands

```bash
# Check DVC status
dvc status --verbose

# Validate configuration
dvc config --list
dvc remote list

# Check pipeline
dvc dag  # Show pipeline DAG
dvc repro --dry  # Show what would be executed

# Check data
dvc list . --recursive
dvc get . data/processed/cogits/ --rev experiment-branch
```

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [DVC Experiment Management](https://dvc.org/doc/user-guide/experiment-management)
- [MLOps Best Practices](https://ml-ops.org/)
- [Reproducible Research Guidelines](https://www.nature.com/articles/s41597-020-0500-y)

## 🎓 Research Citation

When using this pipeline in research, please include:

```bibtex
@software{cogit_manipulation_pipeline,
  title={Cognitive Manipulation Research Pipeline with DVC},
  author={Your Research Team},
  year={2025},
  url={https://github.com/your-org/project-cogit-framework},
  version={1.0}
}
```

---

*This guide ensures reproducible, collaborative, and scientifically rigorous cognitive manipulation research using DVC best practices.*