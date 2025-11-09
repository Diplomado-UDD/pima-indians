# EDA Guide: Understanding the Data Before Modeling

## Why EDA First?

**EDA is NOT optional** - it's the foundation of any trustworthy ML system, especially in healthcare:

1. **Data Quality**: Discover missing values, outliers, errors
2. **Clinical Validity**: Verify features align with medical knowledge
3. **Bias Detection**: Identify underrepresented groups
4. **Feature Understanding**: Know what drives predictions
5. **Preprocessing Justification**: Evidence-based decisions

## Running the EDA Notebook

### Prerequisites

- Python 3.12+
- uv installed
- Dependencies synced (`uv sync`)

### Quick Start

```bash
# Launch marimo notebook
uv run marimo edit notebooks/01_exploratory_data_analysis.py

# This opens a browser with interactive analysis
```

### What the Notebook Contains

#### 1. Data Overview (Section 1)
- Dataset size: 768 samples, 8 features
- Class distribution: 35% diabetic
- Feature descriptions with clinical context

#### 2. Missing Data Analysis (Section 2)
**Critical Finding**:
- Zeros represent missing values (biologically impossible for critical features)
- Insulin: 49% missing (note: zeros may be valid for fasting insulin, not imputed)
- SkinThickness: 30% missing
- BloodPressure: 5% missing
- Glucose: Small % missing (critical feature)
- BMI: Small % missing (critical feature)

**Decision**: Median imputation for Glucose, BloodPressure, SkinThickness, BMI (implemented in pipeline). Insulin zeros are kept as-is.

#### 3. Univariate Analysis (Section 3)
- Distribution of each feature
- Skewness and outliers
- Statistical summaries

**Key Insight**:
- Glucose right-skewed (diabetic population)
- BMI median = 32 (obese range)
- Age range 21-81, median 29 (young cohort)

#### 4. Bivariate Analysis (Section 4)
- How features differ between diabetic/non-diabetic
- Effect sizes for each predictor

**Top Predictors**:
1. Glucose: +29% higher in diabetic group
2. BMI: +14% higher
3. Age: +27% higher

#### 5. Correlation Analysis (Section 5)
- Feature relationships
- Multicollinearity check
- Correlation with diabetes outcome

**Finding**: No severe multicollinearity (all r < 0.8)

#### 6. Clinical Context (Section 6)
- Glucose thresholds: <100 (normal), 100-125 (prediabetes), ≥126 (diabetes)
- BMI categories: <25 (normal), 25-30 (overweight), ≥30 (obese)

**Validation**: Features align with diagnostic criteria

#### 7. Subgroup Analysis (Section 7)
- Diabetes rate by age group
- Diabetes rate by BMI category

**Bias Alert**:
- Younger cohort (limited >60 representation)
- **Action**: Monitor model performance across all ages

#### 8. Outlier Detection (Section 8)
- Z-score based outlier identification
- Outlier prevalence by feature

**Decision**: Keep outliers (valid extreme cases)

#### 9. Key Findings & Recommendations (Section 9)
- Summary of all discoveries
- Preprocessing justification
- Model design rationale

## How EDA Informed the Pipeline

### Data Quality Fixes
```python
# Issue: Zeros as missing values for critical biological features
# Solution: ZeroImputer with median imputation
# Note: Only imputes Glucose, BloodPressure, SkinThickness, BMI
# Insulin zeros are kept as-is (may be valid for fasting insulin)
ZeroImputer(columns=['Glucose', 'BloodPressure', 'SkinThickness', 'BMI'])
```

### Feature Engineering Rationale
```python
# Issue: Non-linear clinical relationships
# Solution: Clinical category features
- BMI_Category (underweight/normal/overweight/obese)
- Age_Group (<30, 30-45, 45-60, >60)
- Glucose_BMI_Interaction
```

### Class Imbalance Strategy
```python
# Issue: 35% positive class
# Solution: Loss-based weighting (preserves calibration)
scale_pos_weight = 3.73  # NOT SMOTE!
```

### Recall Optimization Justification
```python
# Clinical Context: Screening tool (not diagnosis)
# Trade-off: High recall (catch cases) vs. precision (minimize FP)
# Decision: Target recall ≥ 0.85, accept precision ≥ 0.45
```

## EDA Findings Validation in Model Results

| EDA Finding | Model Implementation | Validation |
|-------------|---------------------|------------|
| Glucose strongest predictor | Feature importance: #1 | ✓ Confirmed |
| Missing values in critical features | Median imputation (Glucose, BloodPressure, SkinThickness, BMI) | ✓ Implemented |
| 49% Insulin missing | Not imputed (zero may be valid for fasting insulin) | ⚠️ Design decision |
| Class imbalance (35%) | Class weighting 3.73x | ✓ Implemented |
| Young cohort bias | Subgroup monitoring needed | ⚠️ To monitor |
| Clinical thresholds | Threshold optimization | ✓ Implemented |

## Interactive Features

The marimo notebook is **fully interactive**:

- **Live filtering**: Adjust outlier thresholds
- **Dynamic plots**: Explore different features
- **Real-time stats**: See calculations update
- **Export results**: Save plots and tables

## Common Questions

### Q: Why median imputation instead of deletion?
**A**: Would lose 50% of data (unacceptable). Median preserves distribution better than mean for skewed features.

### Q: Why not use SMOTE for class imbalance?
**A**: SMOTE creates synthetic data that hurts calibration. Medical applications need trustworthy probabilities. Class weighting achieves same recall boost without calibration damage.

### Q: Should I remove outliers?
**A**: No. Extreme values may be real high-risk patients. Tree-based models (LightGBM) handle outliers naturally.

### Q: Is the dataset biased?
**A**: Yes - younger cohort, limited >60 representation. Monitor model performance across age groups in production.

## Next Steps After EDA

1. ✓ **Data validation** (Pandera schema)
2. ✓ **Preprocessing pipeline** (ZeroImputer + FeatureEngineer)
3. ✓ **Model training** (LightGBM with class weighting)
4. ✓ **Evaluation** (Recall-focused metrics)
5. ⚠️ **Subgroup testing** (Performance across age/BMI groups)

## Running EDA as Part of Workflow

### Recommended Workflow

```bash
# 1. Start with EDA
uv run marimo edit notebooks/01_exploratory_data_analysis.py

# 2. Review findings, adjust preprocessing if needed
# Edit: src/features/preprocess.py

# 3. Run data validation
uv run python -m src.data.validate_input data/raw/diabetes.csv

# 4. Train model
uv run python -m src.models.train --config configs/train_config.yaml

# 5. Evaluate (including subgroup analysis)
uv run python -m src.models.evaluate --model-path models/production/model_artifacts.pkl
```

## Exporting EDA Results

```bash
# Run notebook headless and save HTML
uv run marimo run notebooks/01_exploratory_data_analysis.py > reports/eda_report.html

# Export specific plots (from within notebook)
# Click "Download" on any plot
```

## For Production Teams

### Pre-Deployment Checklist

- [ ] EDA notebook reviewed by domain expert (clinician)
- [ ] All data quality issues addressed in preprocessing
- [ ] Subgroup analysis shows no severe performance disparities
- [ ] Clinical thresholds align with medical guidelines
- [ ] Bias mitigation strategy documented

### Monitoring Dashboard

Use EDA as template for production monitoring:
- Track feature distributions (data drift)
- Monitor subgroup performance (fairness)
- Alert on out-of-range values (data quality)

## References

- [Marimo Documentation](https://docs.marimo.io/)
- [Pima Indians Dataset Info](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Medical Screening Guidelines](../docs/ARCHITECTURE.md#clinical-context-analysis)
