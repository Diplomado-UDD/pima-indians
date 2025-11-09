# Technical Architecture

## System Overview

Production-ready MLOps pipeline for diabetes risk prediction optimized for **recall/sensitivity** in medical screening applications.

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
│  - Historical: diabetes.csv (DVC tracked)                   │
│  - Production: Daily batch CSVs                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Exploratory Data Analysis (EDA)                 │
│  - Data quality assessment (missing values, outliers)       │
│  - Clinical context validation (diagnostic thresholds)      │
│  - Bias detection (subgroup analysis)                       │
│  - Feature importance and correlations                      │
│  - Interactive marimo notebook                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Validation & Preprocessing                 │
│  - Schema validation (Pandera)                              │
│  - Zero-value imputation for biological features            │
│  - Feature engineering (BMI categories, interactions)       │
│  - Outlier detection (z-score method)                       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Model Training Pipeline                     │
│  - Algorithm: LightGBM with class weighting                 │
│  - Optimization: F2-score (recall-focused)                  │
│  - Hyperparameter tuning: Optuna (100 trials)              │
│  - Threshold optimization: Precision-recall curve           │
│  - Experiment tracking: MLflow                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Model Registry & Versioning                 │
│  - Storage: Local filesystem (models/production/)           │
│  - Versioning: Semantic (v{major}.{minor}.{patch})         │
│  - Metadata: JSON with metrics, params, timestamps         │
│  - Artifacts: Trained model + preprocessing pipeline       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Batch Inference Service (Docker)                │
│  - Input: CSV with patient features                         │
│  - Processing: Validate → Preprocess → Predict             │
│  - Output: Predictions with risk scores & categories       │
│  - Logging: JSONL for audit trail                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring & Alerting                     │
│  - Data drift: Evidently AI (KS test, chi-squared)         │
│  - Performance: Delayed ground truth evaluation            │
│  - System health: Logs, metrics, throughput               │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 0. Exploratory Data Analysis Layer

**Tool:** Marimo interactive notebook

**Sections:**
1. Data overview (768 samples, 8 features, 35% positive class)
2. Missing data analysis (49% Insulin, 30% SkinThickness missing - note: Insulin not imputed)
3. Univariate distributions (skewness, outliers)
4. Bivariate analysis (features vs. diabetes outcome)
5. Correlation analysis (multicollinearity check)
6. Clinical context validation (diagnostic thresholds)
7. Subgroup/bias analysis (age, BMI groups)
8. Outlier detection (z-score method)
9. Key findings and recommendations

**Key Insights:**
- Glucose strongest predictor (+29% in diabetic group)
- Class imbalance 35% (justifies class weighting, NOT SMOTE)
- No severe multicollinearity (all r < 0.8)
- Younger cohort bias identified → monitoring needed
- All preprocessing decisions validated by evidence

**Running EDA:**
```bash
uv run marimo edit notebooks/01_exploratory_data_analysis.py
```

### 1. Data Layer

**Storage:**
- DVC for dataset versioning (local remote)
- Git for code versioning
- Local filesystem for intermediate/output data

**Schema:**
- Defined in `src/config/constants.py` (REQUIRED_COLUMNS)
- Consistent across validation, preprocessing, and inference
- Schema validation using Pandera with DiabetesDataValidator

```python
{
    "Pregnancies": int (≥0),
    "Glucose": float (≥0),
    "BloodPressure": float (≥0),
    "SkinThickness": float (≥0),
    "Insulin": float (≥0),
    "BMI": float (≥0),
    "DiabetesPedigreeFunction": float (≥0),
    "Age": int (0-120),
    "Outcome": int (0 or 1)  # Training only
}
```

**Preprocessing Pipeline:**
```python
Pipeline([
    ('zero_imputer', ZeroImputer(columns=ZERO_IMPUTE_COLUMNS)),  # From constants
    # ZERO_IMPUTE_COLUMNS = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
    # Note: Insulin is NOT imputed (zero may be valid for fasting insulin)
    ('feature_engineer', FeatureEngineer()),  # Creates derived features
    ('scaler', StandardScaler())  # Standardization
])
```

**Derived Features:**
- `BMI_Category`: 0=Underweight, 1=Normal, 2=Overweight, 3=Obese
- `Age_Group`: 0=<30, 1=30-45, 2=45-60, 3=>60
- `Glucose_BMI_Interaction`: Glucose × BMI

### 2. Training Layer

**Algorithm:** LightGBM Classifier

**Key Hyperparameters (Optimized):**
```yaml
learning_rate: 0.013
max_depth: 5
num_leaves: 64
scale_pos_weight: 3.73  # Penalizes FN 3.7x more than FP
min_child_samples: 25
subsample: 0.996
colsample_bytree: 0.749
```

**Optimization Strategy:**
1. **Objective:** Maximize F2-score (β=2 emphasizes recall)
2. **Constraints:**
   - Recall ≥ 0.85 (must catch 85% of diabetes cases)
   - Precision ≥ 0.45 (avoid >55% false positive rate)
3. **Method:** Optuna TPE sampler, 100 trials
4. **Threshold:** Optimized on precision-recall curve (0.459)

**Training Workflow:**
```python
1. Load data → Stratified split (60/20/20)
2. Fit preprocessor on training set
3. Hyperparameter tuning with Optuna
   - Search space: 7 parameters
   - Metric: F2-score
   - Validation: 5-fold stratified CV
4. Find optimal threshold (maximize precision @ recall≥0.85)
5. Evaluate on held-out test set
6. Save artifacts: model, preprocessor, threshold, metadata
7. Log to MLflow: params, metrics, artifacts
```

### 3. Inference Layer

**Deployment:** Docker container

**Architecture:**
```dockerfile
python:3.12-slim (base)
  ├── uv (dependency manager)
  ├── src/ (application code)
  │   ├── config/ (shared constants)
  │   ├── data/ (validation using DiabetesDataValidator)
  │   ├── inference/ (batch prediction with logging)
  │   └── monitoring/ (drift detection)
  ├── models/production/ (mounted volume)
  └── configs/ (configuration)
```

**Prediction Flow:**
```python
1. Load production model + metadata
2. Validate input CSV schema (using DiabetesDataValidator)
3. Retry logic for file operations (configurable)
4. Apply preprocessing pipeline
5. Perform drift detection (if enabled in config)
6. Generate probabilities: model.predict_proba()
7. Apply optimal threshold: preds = (proba >= 0.459)
8. Categorize risk: Low (<0.35) | Medium (0.35-0.60) | High (>0.60)
9. Save predictions + structured logging (JSONL)
10. Error handling with detailed logging at each step
```

**Output Format:**
```csv
patient_id,screening_recommendation,diabetes_risk_score,risk_category,model_version,prediction_date
P0001,1,0.691,High,20251108_190038,2025-11-08T19:06:49.819798
```

### 4. Monitoring Layer

**Data Drift Detection:**
- **Method:** Evidently AI library
- **Integration:** Automatically runs during batch prediction (if enabled)
- **Tests:**
  - Continuous features: Kolmogorov-Smirnov test
  - Categorical features: Chi-squared test
- **Frequency:** After each batch (configurable)
- **Alert threshold:** Drift in ≥3 features or p-value < 0.05
- **Reporting:** Drift results included in batch processing reports

**Performance Monitoring:**
- **Prediction logging:** Structured logging with Python `logging` module (JSONL format)
- **Log levels:** Configurable (INFO/DEBUG/WARNING/ERROR)
- **Delayed evaluation:** When labels available weeks/months later
- **Metrics tracked:** Recall, precision, F2, AUC, FN count, drift status
- **Alerting:** If recall < 0.80 for 2 consecutive weeks or drift detected

**System Health:**
- Batch processing time
- Record throughput
- Error rates by type (with retry statistics)
- Resource usage (CPU, memory)
- Retry attempt tracking

**Error Handling:**
- Comprehensive try-catch blocks for all operations
- Detailed error messages with context
- Failed operations logged with stack traces (debug mode)
- Rejected batches saved with error reports

### 5. Model Registry

**Directory Structure:**
```
models/
├── production/              # Current production model
│   ├── model_artifacts.pkl  # LightGBM + preprocessor + threshold
│   └── metadata.json        # Version, metrics, params
├── production_backup/       # Previous version (rollback)
├── experiments/             # Training artifacts
│   └── model_YYYYMMDD_HHMMSS/
└── archive/                 # Historical versions
```

**Metadata Schema:**
```json
{
  "version": "20251108_190038",
  "algorithm": "LightGBM",
  "optimization_target": "recall",
  "optimal_threshold": 0.459,
  "target_recall": 0.85,
  "test_metrics": {
    "test_recall": 0.852,
    "test_precision": 0.568,
    "test_f2_score": 0.774,
    "test_false_negatives": 8
  },
  "training_date": "2025-11-08T19:00:38",
  "data_path": "data/raw/diabetes.csv",
  "best_params": {...}
}
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.12 | Core implementation |
| **Dependency Mgmt** | uv | Fast, deterministic package management |
| **ML Framework** | LightGBM | Gradient boosting (recall-optimized) |
| **Optimization** | Optuna | Hyperparameter tuning |
| **Experiment Tracking** | MLflow | Parameter/metric logging, artifact storage |
| **Data Versioning** | DVC | Dataset version control |
| **Validation** | Pandera | Schema validation |
| **Drift Detection** | Evidently AI | Statistical tests for distribution shift |
| **Data Processing** | pandas, scikit-learn | Preprocessing, feature engineering |
| **Containerization** | Docker | Deployment isolation |
| **Testing** | pytest | Unit/integration tests |
| **Code Quality** | black, ruff | Formatting, linting |

## Scalability Considerations

**Current Capacity:**
- **Training:** 768 samples, 100 trials → 30 seconds
- **Inference:** 630 patients/second (single container)
- **Batch size:** Designed for 10K records/day

**Scaling Strategy:**
1. **Horizontal:** Run multiple containers in parallel for large batches
2. **Vertical:** Increase container resources (CPU/memory)
3. **Distributed training:** For larger datasets (>100K samples)
4. **Model optimization:** ONNX conversion for faster inference

**Bottlenecks:**
- I/O (reading large CSVs) → Use Parquet or chunked processing
- Feature engineering → Vectorize operations, use Polars
- Model size → Already small (200KB), not a concern

## Security & Compliance

**Data Protection:**
- No PHI in logs (patient_id only, no names/DOB)
- Local storage (no cloud egress)
- Access controls via filesystem permissions

**Audit Trail:**
- All predictions logged with timestamps
- Model version tracking
- Reproducible predictions (model + input → same output)

**Regulatory Readiness:**
- Model cards documenting intended use, limitations
- Version control for all code and data
- Validation reports for each model version

## Deployment Patterns

**Development:**
```bash
uv run python -m src.inference.batch_predict --input test.csv
```

**Production (Docker):**
```bash
docker run -v /data:/data diabetes-batch-predictor:v1.0 \
    --input /data/incoming/batch.csv
```

**Scheduled (Cron):**
```cron
0 2 * * * docker run --rm -v /data:/data diabetes-batch-predictor:latest \
    --input /data/incoming/batch_$(date +\%Y\%m\%d).csv
```

## Future Enhancements

1. **Real-time API:** FastAPI endpoint for individual predictions
2. **Model ensemble:** Combine multiple models for better performance
3. **Active learning:** Flag uncertain cases for manual review
4. **Explainability:** SHAP values for prediction interpretation
5. **A/B testing:** Shadow deployment for model comparison
6. **Continuous training:** Automated retraining pipeline

## References

- [MLOps Blueprint Design](../README.md)
- [Quick Start Guide](QUICKSTART.md)
- [Model Evaluation Report](../reports/model_evaluation/)
- [MLflow Tracking](../mlruns/)
