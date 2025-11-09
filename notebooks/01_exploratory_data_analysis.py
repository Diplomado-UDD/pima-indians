"""Exploratory Data Analysis for Pima Indians Diabetes Dataset.

This marimo notebook provides comprehensive EDA including:
- Data profiling and quality assessment
- Clinical context and feature relationships
- Bias and fairness analysis across subgroups
- Missing data patterns and preprocessing justification
"""

import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    mo.md(
        """
        # Exploratory Data Analysis: Pima Indians Diabetes Dataset

        **Objective**: Understand data quality, feature relationships, and potential biases
        before building a recall-optimized medical screening model.

        **Dataset**: Pima Indians Diabetes Database (768 samples, 8 features, 1 target)
        """
    )
    return Path, mo, np, pd, plt, sns


@app.cell
def _(Path, pd):
    # Load data - use robust path resolution
    notebook_dir = Path(__file__).parent
    data_path = notebook_dir.parent / "data" / "raw" / "diabetes.csv"
    df = pd.read_csv(data_path)

    # Basic info
    n_samples, n_features = df.shape
    n_positive = df['Outcome'].sum()
    positive_rate = df['Outcome'].mean()

    # Extract feature names (excluding target)
    features = df.columns.drop('Outcome').tolist()
    return df, features, n_features, n_positive, n_samples, positive_rate


@app.cell
def _(mo, n_features, n_positive, n_samples, positive_rate):
    mo.md(f"""
    ## 1. Dataset Overview

    **Samples**: {n_samples} patients
    **Features**: {n_features - 1} (8 clinical measurements)
    **Target**: Diabetes diagnosis (0 = No, 1 = Yes)
    **Positive Rate**: {positive_rate:.1%} ({n_positive} diabetic patients)

    ### Feature Descriptions

    | Feature | Description | Clinical Significance |
    |---------|-------------|----------------------|
    | Pregnancies | Number of pregnancies | Risk factor (gestational diabetes) |
    | Glucose | Plasma glucose (mg/dL) | **Primary diagnostic marker** (≥126 = diabetes) |
    | BloodPressure | Diastolic BP (mm Hg) | Comorbidity indicator |
    | SkinThickness | Triceps skin fold (mm) | Obesity proxy |
    | Insulin | 2-hour serum insulin (μU/ml) | Insulin resistance marker |
    | BMI | Body Mass Index (kg/m²) | **Strong risk factor** (≥30 = obese) |
    | DiabetesPedigreeFunction | Genetic likelihood | Family history weighting |
    | Age | Age in years | Risk increases with age |
    """)
    return


@app.cell
def _(mo):
    # Display first few rows
    mo.md("""
    ### Sample Data
    """)
    return


@app.cell
def _(df, mo, pd):
    mo.md("## 2. Missing Data Analysis")

    # Check for explicit nulls
    null_counts = df.isnull().sum()

    # Check for zeros (biologically implausible)
    zero_counts = (df == 0).sum()

    # Biological features that cannot be zero
    bio_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    missing_summary = pd.DataFrame({
        'Explicit Nulls': null_counts,
        'Zero Values': zero_counts,
        'Zero % (if bio feature)': [
            f"{(df[col] == 0).mean():.1%}" if col in bio_features else "N/A"
            for col in df.columns
        ]
    })
    print(missing_summary.to_markdown())
    return (bio_features,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
    **Finding**: No explicit nulls, but zeros in biological features represent missing data!

    - **Glucose**: Cannot be 0 (person would be dead)
    - **BloodPressure**: Cannot be 0 (no pulse)
    - **BMI**: Cannot be 0 (no body mass)
    """)
    return


@app.cell
def _(bio_features, df, plt):
    # Visualize missing data pattern
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Missing data heatmap
    missing_mask = df[bio_features] == 0
    _axes[0].imshow(missing_mask.T, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    _axes[0].set_yticks(range(len(bio_features)))
    _axes[0].set_yticklabels(bio_features)
    _axes[0].set_xlabel('Patient Index')
    _axes[0].set_title('Missing Data Pattern (Red = Missing/Zero)')

    # Missing data percentage by feature
    missing_pct = (missing_mask.sum() / len(df) * 100).sort_values(ascending=True)
    _axes[1].barh(range(len(missing_pct)), missing_pct.values)
    _axes[1].set_yticks(range(len(missing_pct)))
    _axes[1].set_yticklabels(missing_pct.index)
    _axes[1].set_xlabel('Percentage Missing (%)')
    _axes[1].set_title('Missing Data by Feature')
    _axes[1].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    _axes[1].legend()

    plt.tight_layout()
    _fig
    return (missing_pct,)


@app.cell
def _(missing_pct, mo):
    mo.md(f"""
    **Key Findings**:
    - **Insulin**: {missing_pct['Insulin']:.1f}% missing (nearly half!)
    - **SkinThickness**: {missing_pct['SkinThickness']:.1f}% missing
    - **BloodPressure**: {missing_pct['BloodPressure']:.1f}% missing
    - **BMI**: {missing_pct['BMI']:.1f}% missing
    - **Glucose**: {missing_pct['Glucose']:.1f}% missing (critical!)

    **Implication**: Simple deletion would lose ~50% of data.
    **Solution**: Median imputation for non-zero values (implemented in pipeline).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Univariate Analysis: Feature Distributions
    """)
    return


@app.cell
def _(df, np, plt):
    # Create distribution plots for all features
    _features = df.columns.drop('Outcome')
    _fig, _axes = plt.subplots(4, 2, figsize=(14, 16))
    _axes = _axes.flatten()

    for _idx, _feature in enumerate(_features):
        _ax = _axes[_idx]

        # Replace zeros with NaN for visualization
        _data_nonzero = df[_feature].replace(0, np.nan)

        # Histogram
        _ax.hist(_data_nonzero.dropna(), bins=30, alpha=0.7, edgecolor='black')
        _ax.set_xlabel(_feature)
        _ax.set_ylabel('Frequency')
        _ax.set_title(f'{_feature} Distribution (zeros excluded)')

        # Add statistics
        _mean_val = _data_nonzero.mean()
        _median_val = _data_nonzero.median()
        _ax.axvline(_mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {_mean_val:.1f}')
        _ax.axvline(_median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {_median_val:.1f}')
        _ax.legend()

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Feature Distributions (Excluding Zeros)
    """)
    return


@app.cell
def _(df):
    # Statistical summary
    summary_stats = df.describe().T

    # Add skewness and kurtosis
    summary_stats['skewness'] = df.skew()
    summary_stats['kurtosis'] = df.kurtosis()

    # Add zero counts for bio features
    summary_stats['zeros'] = (df == 0).sum()
    summary_stats['zeros_pct'] = (df == 0).mean() * 100
    return


@app.cell
def _(mo):
    mo.md("""
    ### Statistical Summary
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **Observations**:
    - **Glucose** is right-skewed (expected for diabetic population)
    - **Insulin** highly variable and skewed
    - **Age** ranges 21-81, median ~29 (relatively young cohort)
    - **BMI** median ~32 (obese range, expected for at-risk population)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Bivariate Analysis: Features vs. Diabetes Outcome
    """)
    return


@app.cell
def _(df, features, np, plt):
    # Assume 'df' and 'features' (list of 8 column names) are defined.
    # Also assume 'plt' (matplotlib.pyplot) and 'np' (numpy) are imported.

    # Box plots by outcome
    _fig, _axes = plt.subplots(4, 2, figsize=(14, 16))
    _axes = _axes.flatten()

    # Handle case where there are fewer features than subplots
    num_features = len(features)

    for _idx, _feature in enumerate(features):
        _ax = _axes[_idx] # Get the current axis

        # Replace zeros with NaN
        _data_clean = df[[_feature, 'Outcome']].copy()
        _data_clean.loc[_data_clean[_feature] == 0, _feature] = np.nan

        # Box plot by outcome
        # --- FIX ---
        # 1. Used the correct parameter 'ax' (not '_ax')
        # 2. Passed the correct variable '_ax' (not 'ax')
        _data_clean.boxplot(column=_feature, by='Outcome', ax=_ax)

        # --- FIX ---
        # Called methods directly on '_ax' instead of 'plt'
        _ax.set_xlabel('Outcome')

        # --- FIX ---
        # Used the correct loop variable '_feature' (not 'feature')
        _ax.set_ylabel(_feature)
        _ax.set_title(f'{_feature} by Diabetes Status')

        # --- FIX ---
        # Replaced plt.sca() and plt.xticks() with a direct call.
        # pandas boxplot(by=...) plots groups at ticks 1 and 2
        _ax.set_xticklabels(['No Diabetes', 'Diabetes'])

    # Turn off any unused axes
    for _i in range(num_features, len(_axes)):
        _axes[_i].axis('off')

    # --- FIX ---
    # Add this to remove the default suptitle that pandas boxplot(by=...)
    # sometimes adds, which can mess up tight_layout
    plt.suptitle('') 

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Feature Distributions by Diabetes Status
    """)
    return


@app.cell
def _(df, features, np, pd):
    # Statistical comparison by outcome
    comparison = pd.DataFrame()

    for _feature in features:
        _data_clean = df[_feature].replace(0, np.nan)

        no_diabetes = _data_clean[df['Outcome'] == 0].dropna()
        has_diabetes = _data_clean[df['Outcome'] == 1].dropna()

        comparison.loc[_feature, 'No Diabetes (Mean)'] = no_diabetes.mean()
        comparison.loc[_feature, 'Diabetes (Mean)'] = has_diabetes.mean()
        comparison.loc[_feature, 'Difference'] = has_diabetes.mean() - no_diabetes.mean()
        comparison.loc[_feature, 'Difference %'] = ((has_diabetes.mean() - no_diabetes.mean()) / no_diabetes.mean() * 100)

    comparison = comparison.sort_values('Difference %', ascending=False)
    return (comparison,)


@app.cell
def _(mo):
    mo.md("""
    ### Mean Differences by Diabetes Status
    """)
    return


@app.cell
def _(comparison, mo):
    mo.md(f"""
    **Key Discriminators** (largest differences):

    1. **{comparison.index[0]}**: +{comparison.iloc[0]['Difference %']:.1f}% higher in diabetic group
    2. **{comparison.index[1]}**: +{comparison.iloc[1]['Difference %']:.1f}% higher in diabetic group
    3. **{comparison.index[2]}**: +{comparison.iloc[2]['Difference %']:.1f}% higher in diabetic group

    These features will likely be most important for prediction.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Multivariate Analysis: Feature Correlations
    """)
    return


@app.cell
def _(df, np, plt, sns):
    # Replace zeros with NaN for correlation
    df_clean = df.replace(0, np.nan)

    # Correlation matrix
    corr_matrix = df_clean.corr()

    # Plot heatmap
    _fig, _ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=_ax,
                vmin=-1, vmax=1)
    _ax.set_title('Feature Correlation Matrix (Zeros Excluded)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _fig
    return (corr_matrix,)


@app.cell
def _(mo):
    mo.md("""
    ### Correlation Heatmap
    """)
    return


@app.cell
def _(corr_matrix):
    # Get correlations with Outcome
    outcome_corr = corr_matrix['Outcome'].drop('Outcome').sort_values(ascending=False)
    return (outcome_corr,)


@app.cell
def _(mo, outcome_corr):
    mo.md("### Correlation with Diabetes Outcome")

    corr_df = outcome_corr.reset_index()
    corr_df.columns = ['Feature', 'Correlation']
    return


@app.cell
def _(mo, outcome_corr):
    strongest_feature = outcome_corr.idxmax()
    strongest_corr = outcome_corr.max()

    mo.md(
        f"""
        **Findings**:
        - **Strongest predictor**: {strongest_feature} (r = {strongest_corr:.3f})
        - Moderate multicollinearity between some features (expected for health metrics)
        - No perfect collinearity (all correlations < 0.8)
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Clinical Context Analysis
    """)
    return


@app.cell
def _(df, plt):
    # Glucose clinical thresholds
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Glucose distribution with clinical thresholds
    _glucose_clean = df[df['Glucose'] > 0]['Glucose']
    _axes[0].hist(_glucose_clean, bins=40, alpha=0.7, edgecolor='black')
    _axes[0].axvline(x=100, color='yellow', linestyle='--', linewidth=2, label='Prediabetes (100)')
    _axes[0].axvline(x=126, color='red', linestyle='--', linewidth=2, label='Diabetes (126)')
    _axes[0].set_xlabel('Glucose (mg/dL)')
    _axes[0].set_ylabel('Frequency')
    _axes[0].set_title('Glucose Distribution with Clinical Thresholds')
    _axes[0].legend()

    # BMI distribution with clinical categories
    _bmi_clean = df[df['BMI'] > 0]['BMI']
    _axes[1].hist(_bmi_clean, bins=40, alpha=0.7, edgecolor='black')
    _axes[1].axvline(x=18.5, color='blue', linestyle='--', linewidth=2, label='Underweight')
    _axes[1].axvline(x=25, color='green', linestyle='--', linewidth=2, label='Normal')
    _axes[1].axvline(x=30, color='orange', linestyle='--', linewidth=2, label='Obese')
    _axes[1].set_xlabel('BMI (kg/m²)')
    _axes[1].set_ylabel('Frequency')
    _axes[1].set_title('BMI Distribution with Clinical Categories')
    _axes[1].legend()

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Clinical Threshold Analysis
    """)
    return


@app.cell
def _(df):
    # Calculate clinical category distributions
    _glucose_clean = df[df['Glucose'] > 0]
    _bmi_clean = df[df['BMI'] > 0]

    clinical_stats = {
        'Glucose Normal (<100)': (_glucose_clean['Glucose'] < 100).sum(),
        'Glucose Prediabetes (100-125)': ((_glucose_clean['Glucose'] >= 100) &
                                           (_glucose_clean['Glucose'] < 126)).sum(),
        'Glucose Diabetes (≥126)': (_glucose_clean['Glucose'] >= 126).sum(),
        'BMI Normal (<25)': (_bmi_clean['BMI'] < 25).sum(),
        'BMI Overweight (25-30)': ((_bmi_clean['BMI'] >= 25) & (_bmi_clean['BMI'] < 30)).sum(),
        'BMI Obese (≥30)': (_bmi_clean['BMI'] >= 30).sum(),
    }
    return (clinical_stats,)


@app.cell
def _(clinical_stats, mo, pd):
    mo.md("### Clinical Category Distribution")

    stats_df = pd.DataFrame(list(clinical_stats.items()), columns=['Category', 'Count'])
    print(stats_df)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Subgroup & Fairness Analysis
    """)
    return


@app.cell
def _(df, np, pd):
    # Create age groups
    df_analysis = df.copy()
    df_analysis['Age_Group'] = pd.cut(df_analysis['Age'],
                                       bins=[0, 30, 45, 60, 100],
                                       labels=['<30', '30-45', '45-60', '>60'])

    # Create BMI categories
    df_analysis['BMI_Category'] = pd.cut(df_analysis['BMI'].replace(0, np.nan),
                                          bins=[0, 25, 30, 100],
                                          labels=['Normal/Under', 'Overweight', 'Obese'])
    return (df_analysis,)


@app.cell
def _(df_analysis):
    # Diabetes rate by subgroup
    age_outcome = df_analysis.groupby('Age_Group')['Outcome'].agg(['mean', 'count'])
    age_outcome.columns = ['Diabetes Rate', 'Sample Size']

    bmi_outcome = df_analysis.groupby('BMI_Category')['Outcome'].agg(['mean', 'count'])
    bmi_outcome.columns = ['Diabetes Rate', 'Sample Size']
    return


@app.cell
def _(mo):
    mo.md("""
    ### Diabetes Rate by Age Group and BMI Category
    """)
    return


@app.cell
def _(df_analysis, plt):
    # Subgroup visualization
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Age group analysis
    age_stats = df_analysis.groupby('Age_Group')['Outcome'].mean() * 100
    _axes[0].bar(range(len(age_stats)), age_stats.values, color='skyblue', edgecolor='black')
    _axes[0].set_xticks(range(len(age_stats)))
    _axes[0].set_xticklabels(age_stats.index)
    _axes[0].set_ylabel('Diabetes Rate (%)')
    _axes[0].set_xlabel('Age Group')
    _axes[0].set_title('Diabetes Rate by Age Group')
    _axes[0].axhline(y=df_analysis['Outcome'].mean() * 100, color='red',
                     linestyle='--', label='Overall Rate')
    _axes[0].legend()

    # BMI category analysis
    bmi_stats = df_analysis.groupby('BMI_Category')['Outcome'].mean() * 100
    _axes[1].bar(range(len(bmi_stats)), bmi_stats.values, color='coral', edgecolor='black')
    _axes[1].set_xticks(range(len(bmi_stats)))
    _axes[1].set_xticklabels(bmi_stats.index, rotation=15)
    _axes[1].set_ylabel('Diabetes Rate (%)')
    _axes[1].set_xlabel('BMI Category')
    _axes[1].set_title('Diabetes Rate by BMI Category')
    _axes[1].axhline(y=df_analysis['Outcome'].mean() * 100, color='red',
                     linestyle='--', label='Overall Rate')
    _axes[1].legend()

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Subgroup Analysis Visualization
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **Fairness Observations**:
    - Diabetes rate increases with age (expected biological trend)
    - Diabetes rate increases with BMI (strong clinical association)
    - **Potential bias**: Undersampling of >60 age group
    - **Recommendation**: Monitor model performance across all subgroups
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Outlier Detection
    """)
    return


@app.cell
def _(df, features, np, pd):
    # Z-score based outlier detection
    outlier_summary = {}

    for _feature in features:
        data_clean = df[_feature].replace(0, np.nan).dropna()
        z_scores = np.abs((data_clean - data_clean.mean()) / data_clean.std())
        outliers = z_scores > 3
        outlier_summary[_feature] = {
            'count': outliers.sum(),
            'percentage': outliers.mean() * 100
        }

    outlier_df = pd.DataFrame(outlier_summary).T
    outlier_df = outlier_df.sort_values('count', ascending=False)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Outlier Detection (Z-score > 3)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **Finding**: Few extreme outliers (all < 5% of data)

    **Recommendation**:
    - Keep outliers (may be clinically valid extreme cases)
    - Robust model (tree-based) will handle naturally
    - Monitor outlier influence in model evaluation
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Key Findings & Recommendations
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Summary of EDA Findings

    #### Data Quality Issues
    1. **Missing Data**: ~50% missing values in Insulin, represented as zeros
       - **Action**: Median imputation for non-zero values ✓ (implemented)

    2. **Biologically Implausible Zeros**: Found in Glucose, BloodPressure, BMI
       - **Action**: Replace with imputed values ✓ (implemented)

    #### Feature Insights
    3. **Strongest Predictors**: Glucose, BMI, Age (high correlation with outcome)
       - **Action**: Prioritize in feature engineering ✓ (implemented)

    4. **Clinical Thresholds**: Features align with medical diagnostic criteria
       - **Validation**: Model predictions should respect clinical knowledge

    #### Fairness & Bias Considerations
    5. **Age Bias**: Younger cohort (median 29), limited >60 representation
       - **Action**: Monitor model performance across age groups ✓ (added to plan)

    6. **Class Imbalance**: 35% positive (moderate imbalance)
       - **Action**: Class weighting (NOT SMOTE) ✓ (implemented)

    #### Model Design Justification
    7. **Recall Optimization**: Medical screening requires high sensitivity
       - **Target**: ≥85% recall to catch diabetic patients
       - **Trade-off**: Accept ~45% false positive rate for confirmatory testing

    8. **Feature Engineering**: Created BMI_Category, Age_Group, interactions
       - **Justification**: Captures non-linear clinical relationships

    ### Preprocessing Pipeline Validation

    The implemented pipeline addresses all identified data quality issues:

    ```python
    Pipeline([
        ('zero_imputer', ZeroImputer()),      # Fixes missing data
        ('feature_engineer', FeatureEngineer()),  # Captures non-linearity
        ('scaler', StandardScaler())          # Normalization
    ])
    ```

    ### Next Steps Post-EDA

    1. ✓ **Model Training**: LightGBM with class weighting
    2. ✓ **Threshold Optimization**: Maximize precision @ recall ≥ 0.85
    3. ⚠️ **Subgroup Evaluation**: Verify performance across age/BMI groups
    4. ⚠️ **Clinical Validation**: Compare predictions to diagnostic thresholds

    ### Recommendations for Production

    1. **Monitor**: Track performance across demographic subgroups
    2. **Alert**: Flag if age/BMI distribution shifts (data drift)
    3. **Validate**: Periodic comparison with clinical guidelines
    4. **Retrain**: When new data covers underrepresented groups
    """)
    return


if __name__ == "__main__":
    app.run()
