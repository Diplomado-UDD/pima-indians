# Pima Indians Diabetes Prediction Pipeline

Production-ready MLOps pipeline for diabetes screening, optimized for **recall/sensitivity** to minimize missed diagnoses.

## Overview

This pipeline implements a binary classification model for diabetes prediction with:
- **Comprehensive EDA**: Interactive analysis of data quality, clinical validity, and bias
- **Recall-optimized training**: Targets 85%+ sensitivity to catch high-risk patients
- **Batch processing**: Daily screening of patient cohorts (up to 10K records)
- **Full observability**: Model performance monitoring, data drift detection, audit trails
- **Reproducibility**: DVC for data, MLflow for experiments, uv for dependencies

## Quick Start

### Prerequisites

- Python 3.12+
- uv (Python package manager)
- Docker (for deployment)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd pima-indians

# Install dependencies
uv sync

# Pull data from DVC
uv run dvc pull
```

### Exploratory Data Analysis

**Start here** - EDA is critical for understanding data quality, clinical validity, and potential biases:

```bash
# Launch interactive marimo notebook
uv run marimo edit notebooks/01_exploratory_data_analysis.py

# Opens browser with 9 comprehensive sections:
# - Data quality assessment (missing values, outliers)
# - Clinical context validation (diagnostic thresholds)
# - Bias detection (age/BMI subgroups)
# - Feature correlations and importance
```

See [EDA Guide](docs/EDA_GUIDE.md) for detailed findings and how they inform the pipeline.

### Training a Model

```bash
# Train recall-optimized model with hyperparameter tuning
uv run python -m src.models.train --config configs/train_config.yaml

# View MLflow UI to track experiments
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# Navigate to http://localhost:5000
```

### Evaluating a Model

```bash
# Generate comprehensive evaluation report
uv run python -m src.models.evaluate \
    --model-path models/experiments/model_<timestamp>/model_artifacts.pkl \
    --test-data data/raw/diabetes.csv \
    --output-dir reports/model_evaluation
```

### Running Batch Predictions

```bash
# Prepare input data
cp <your-batch-file.csv> data/incoming/batch.csv

# Run batch prediction
uv run python -m src.inference.batch_predict \
    --input data/incoming/batch.csv \
    --model-dir models/production \
    --config configs/inference_config.yaml

# Check predictions
cat data/predictions/predictions_batch_*.csv
```

### Docker Deployment

```bash
# Build Docker image
docker build -t diabetes-batch-predictor:latest .

# Run batch processing via Docker
docker run --rm \
    -v $(pwd)/data:/data \
    -v $(pwd)/logs:/logs \
    -v $(pwd)/models/production:/app/models/production:ro \
    diabetes-batch-predictor:latest \
    --input /data/incoming/batch.csv \
    --model-dir /app/models/production
```

## Project Structure

```
pima-indians/
├── configs/                    # Configuration files
│   ├── train_config.yaml       # Training hyperparameters
│   └── inference_config.yaml   # Batch prediction settings
├── data/
│   ├── raw/                    # Original datasets (DVC-tracked)
│   ├── incoming/               # Daily batch inputs
│   ├── predictions/            # Batch prediction outputs
│   └── rejected/               # Invalid inputs
├── notebooks/
│   └── 01_exploratory_data_analysis.py  # Interactive EDA (marimo)
├── src/
│   ├── data/                   # Data validation
│   ├── features/               # Preprocessing & feature engineering
│   ├── models/                 # Training & evaluation
│   ├── inference/              # Batch prediction
│   ├── monitoring/             # Drift detection & metrics
│   └── audit/                  # Audit trail verification
├── models/
│   ├── production/             # Current production model
│   └── experiments/            # Training experiment artifacts
├── tests/                      # Unit & integration tests
├── Dockerfile                  # Production container
└── docker-compose.yml          # Local testing
```

## Key Features

### Exploratory Data Analysis

Evidence-based ML starts with thorough data understanding:
- **Data quality**: 49% Insulin missing, 30% SkinThickness missing (zeros = missing)
- **Clinical validation**: Features align with diagnostic criteria (Glucose ≥126 = diabetes)
- **Bias detection**: Younger cohort identified, performance monitoring across age groups
- **Feature insights**: Glucose strongest predictor (+29% in diabetic group)
- **Interactive analysis**: Marimo notebook with 9 comprehensive sections

All preprocessing and model design decisions justified by EDA findings.

### Recall Optimization

Model is tuned to maximize **sensitivity (recall)** while maintaining acceptable precision:
- **Target**: Recall ≥ 0.85 (catch 85%+ of diabetes cases)
- **Constraint**: Precision ≥ 0.45 (avoid overwhelming clinics)
- **Method**: F2-score optimization + custom threshold tuning

### Reproducibility

- **Data**: DVC tracks dataset versions
- **Code**: Git version control
- **Environment**: `uv.lock` ensures exact dependency versions
- **Experiments**: MLflow logs all parameters, metrics, artifacts
- **Seeds**: Fixed random seeds for deterministic results

### Monitoring

- **Prediction logging**: All predictions logged with model version, timestamp
- **Data drift detection**: Evidently AI monitors feature distributions
- **Performance tracking**: Delayed ground truth evaluation (when labels available)
- **Audit trails**: Reproduce any historical prediction

## Model Performance (Example)

*Note: Update with actual metrics after training*

| Metric | Value |
|--------|-------|
| Recall (Sensitivity) | 0.87 |
| Precision | 0.51 |
| F2-Score | 0.78 |
| AUC-ROC | 0.82 |
| Optimal Threshold | 0.35 |

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### Code Formatting

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/
```

### Adding New Dependencies

```bash
# Add runtime dependency
uv add <package-name>

# Add dev dependency
uv add --dev <package-name>
```

## Configuration

### Training Configuration (`configs/train_config.yaml`)

Key parameters:
- `optimization.target_metric`: "f2_score" (emphasizes recall)
- `optimization.min_recall_constraint`: 0.85
- `hyperparameter_tuning.n_trials`: 100

### Inference Configuration (`configs/inference_config.yaml`)

Key settings:
- `validation.required_columns`: List of expected features
- `output.risk_thresholds`: Low/Medium/High risk categories
- `monitoring.enable_drift_detection`: true

## Deployment

### Local Deployment

Use cron or systemd timer to schedule daily batch processing:

```bash
# Example cron entry (runs daily at 2 AM)
0 2 * * * cd /path/to/pima-indians && uv run python -m src.inference.batch_predict --input data/incoming/batch_$(date +\%Y\%m\%d).csv
```

### Production Deployment (Docker)

```bash
# Build and tag
docker build -t diabetes-batch-predictor:v1.0.0 .

# Push to registry (DockerHub example)
docker tag diabetes-batch-predictor:v1.0.0 <username>/diabetes-batch-predictor:v1.0.0
docker push <username>/diabetes-batch-predictor:v1.0.0

# Deploy on production server
docker pull <username>/diabetes-batch-predictor:v1.0.0
docker run --rm -v /data:/data -v /logs:/logs diabetes-batch-predictor:v1.0.0 ...
```

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. Run tests: `uv run pytest`
4. Format code: `uv run black src/ tests/`
5. Commit: `git commit -m "feat: your feature description"`
6. Push and create pull request

## License

[Specify license]

## Contact

[Specify contact information]
