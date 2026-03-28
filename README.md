# CasCrop: Crop Waste as Economic Contagion

A graph neural network approach to predicting agricultural loss cascades.

## The Problem

Billions of dollars in US crops go to waste annually — not from drought or frost, but from economic forces. When a bumper harvest in Iowa crashes corn prices, farmers in Indiana may abandon perfectly healthy crops because harvesting costs more than the depressed market price. This economic contagion propagates across regions like a financial crisis spreads through banks.

Every existing crop prediction model treats each county independently. None capture these inter-regional economic dependencies. CasCrop fills that gap.

## What's Novel

1. **Asymmetric Economic Contagion Message Passing (ECMP)**: Graph attention where positive and negative price shocks have separate learned transformations. Price drops (oversupply) cause waste contagion; price spikes (undersupply) reduce it.

2. **Biophysical-Economic Disentanglement**: Adversarial training forces the model to separate weather-driven features from market-driven features, proving the graph captures true economic contagion rather than shared weather patterns.

3. **Crop waste (not yield) prediction**: First model to predict whether crops will be wasted using insurance claims as training labels.

## Quick Start

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run full pipeline
python scripts/09_full_pipeline.py

# Or run steps individually
python scripts/01_download_data.py    # Download datasets (~2-4h)
python scripts/02_process_data.py     # Process & match (~1-2h)
python scripts/03_build_graphs.py     # Build graphs (~30min)
python scripts/04_train_all.py        # Train models (~20-40h GPU)
python scripts/05_evaluate_all.py     # Evaluate (~2h)
python scripts/06_case_study.py       # Case study (~1h)
python scripts/07_generate_figures.py # Figures (~30min)
python scripts/08_generate_tables.py  # Tables (~15min)
```

## Docker

```bash
docker build -t cascrop .
docker run --gpus all cascrop
```

## Project Structure

```
cascrop/
├── configs/          # YAML configs for all ablation experiments
├── src/
│   ├── data/         # Data loaders, processing, graph construction
│   ├── models/       # CasCrop + all baselines
│   ├── training/     # Training loop, losses, schedulers
│   ├── evaluation/   # Metrics, statistical tests, case study
│   └── visualization/# Figures, maps, tables (LaTeX)
├── scripts/          # Pipeline scripts (01-09)
├── paper/            # LaTeX manuscript + figures + tables
├── patent/           # Provisional patent application
└── tests/            # Unit tests
```

## Ablation Table

| Model | AUC-ROC | Description |
|-------|---------|-------------|
| Row 1: Local Only | baseline | MLP on biophysical features |
| Row 2: Local + Econ | +econ | Adds economic features |
| Row 3: Geo GAT | +graph | Standard GAT on geographic adjacency |
| Row 4: Symmetric ECMP | +shock | Shared shock transformation |
| **Row 5: CasCrop** | **best** | **Asymmetric ECMP + disentanglement** |

## Data Sources

All freely available:
- USDA RMA crop insurance claims (labels)
- USDA NASS crop production statistics
- Sentinel-2 / Landsat vegetation indices (via Google Earth Engine)
- NOAA weather data
- FRED commodity futures prices
- Census Bureau county adjacency

## Citation

```bibtex
@article{krishnan2026cascrop,
  title={Crop Waste as Economic Contagion: A Graph Neural Network Approach
         to Predicting Agricultural Loss Cascades},
  author={Krishnan, Keshav},
  year={2026}
}
```

## License

Apache 2.0
