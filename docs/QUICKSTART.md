# Quick Start Guide

## Prerequisites

- Python 3.12+
- uv installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker (for deployment)
- Git

## Initial Setup

```bash
# Clone repository
cd pima-indians

# Install dependencies
uv sync

# Pull data from DVC
uv run dvc pull
```

## Exploratory Data Analysis

**Start here** before training:

```bash
# Launch interactive marimo notebook
uv run marimo edit notebooks/01_exploratory_data_analysis.py
# Opens in browser with 9 comprehensive sections
```

See [EDA_GUIDE.md](EDA_GUIDE.md) for findings and methodology.

## Training a Model

```bash
# Train with config (recall-optimized)
uv run python -m src.models.train --config configs/train_config.yaml

# View experiments in MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# Open browser to http://localhost:5000
```

**Expected output:**
- Model saved to `models/experiments/model_<timestamp>/`
- Recall ≥ 0.85, Precision ≥ 0.45

## Evaluating a Model

```bash
# Generate evaluation report
uv run python -m src.models.evaluate \
    --model-path models/experiments/model_<timestamp>/model_artifacts.pkl \
    --output-dir reports/model_evaluation

# View results
open reports/model_evaluation/confusion_matrix.png
open reports/model_evaluation/pr_curve.png
```

## Deploying to Production

```bash
# Copy best model to production
cp -r models/experiments/model_<best_timestamp>/* models/production/

# Verify deployment
ls models/production/
# Should see: model_artifacts.pkl, metadata.json
```

## Running Batch Predictions

### Prepare Input Data

Your CSV must have these columns + optional `patient_id`:
```
patient_id,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
P0001,6,148,72,35,0,33.6,0.627,50
P0002,1,85,66,29,0,26.6,0.351,31
```

### Run Prediction

```bash
# Place input in data/incoming/
cp your_batch.csv data/incoming/

# Run batch prediction
uv run python -m src.inference.batch_predict \
    --input data/incoming/your_batch.csv \
    --model-dir models/production \
    --config configs/inference_config.yaml

# Check results
cat data/predictions/predictions_your_batch_*.csv
```

**Features:**
- **Automatic validation**: Uses DiabetesDataValidator for schema validation
- **Drift detection**: Automatically runs if enabled in config
- **Retry logic**: Configurable retries for file operations
- **Structured logging**: All operations logged with configurable log levels
- **Error handling**: Comprehensive error handling with detailed messages

**Output columns:**
- `patient_id`: Original ID
- `screening_recommendation`: 0 or 1 (binary decision)
- `diabetes_risk_score`: 0.0-1.0 (probability)
- `risk_category`: Low/Medium/High
- `model_version`: Tracking
- `prediction_date`: Timestamp

## Docker Deployment

```bash
# Build image
docker build -t diabetes-batch-predictor:v1.0 .

# Run batch job
docker run --rm \
    -v $(pwd)/data:/data \
    -v $(pwd)/models/production:/app/models/production:ro \
    diabetes-batch-predictor:v1.0 \
    --input /data/incoming/batch.csv
```

## Monitoring for Data Drift

**Automatic Drift Detection:**
Drift detection is automatically integrated into batch prediction when enabled in `configs/inference_config.yaml`:
```yaml
monitoring:
    enable_drift_detection: true
    reference_data_path: "data/raw/diabetes.csv"
    drift_report_dir: "reports/drift"
```

Drift results are included in the batch processing report and logged as warnings if detected.

**Manual Drift Detection:**
```bash
# Compare production data to training data manually
uv run python -m src.monitoring.drift_detection \
    --reference data/raw/diabetes.csv \
    --current data/incoming/recent_batch.csv \
    --output-dir reports/drift

# View HTML report
open reports/drift/drift_report_*.html
```

**If drift detected:**
- Investigate data quality
- Consider retraining model
- Update feature preprocessing if needed
- Check logs for drift details in batch processing reports

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Ensure you're in virtual environment
uv sync
```

### Issue: DVC data not found

```bash
# Pull data from DVC
uv run dvc pull
```

### Issue: MLflow database locked

```bash
# Stop any running MLflow servers
pkill -f mlflow
```

### Issue: Docker build fails

```bash
# Check Docker daemon is running
docker ps

# Clean build cache
docker system prune -a
```

### Issue: Batch prediction fails with validation errors

```bash
# Check input data schema matches required columns
# Required columns are defined in src/config/constants.py
# Or check configs/inference_config.yaml validation.required_columns

# View detailed error logs
uv run python -m src.inference.batch_predict \
    --input data/incoming/your_batch.csv \
    --config configs/inference_config.yaml
# Check logs/predictions/ for detailed error messages
```

### Issue: Drift detection not running

```bash
# Ensure drift detection is enabled in config
# Check configs/inference_config.yaml:
# monitoring.enable_drift_detection: true

# Verify reference data path exists
ls data/raw/diabetes.csv
```

## Performance Benchmarks

- **Training**: ~30 seconds (100 Optuna trials, 768 samples)
- **Batch prediction**: ~30ms for 19 records
- **Throughput**: ~630 patients/second
- **Model size**: ~200KB

## Next Steps

1. **Production deployment**: Use `scripts/deploy.sh`
2. **Schedule batch jobs**: Set up cron/systemd timers
3. **Monitor performance**: Track recall on labeled data
4. **Retrain quarterly**: Or when drift detected
