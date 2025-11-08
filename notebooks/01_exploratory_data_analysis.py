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
def __():
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
    return mo, pd, np, plt, sns, Path


@app.cell
def __(pd, Path):
    # Load data
    data_path = Path("../data/raw/diabetes.csv")
    df = pd.read_csv(data_path)

    # Basic info
    n_samples, n_features = df.shape
    n_positive = df['Outcome'].sum()
    positive_rate = df['Outcome'].mean()

    return df, n_samples, n_features, n_positive, positive_rate, data_path


@app.cell
def __(mo, df, n_samples, n_features, n_positive, positive_rate):
    mo.md(
        f"""
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
        """
    )
    return


@app.cell
def __(mo, df):
    # Display first few rows
    mo.md("### Sample Data")
    return mo.ui.table(df.head(10))


@app.cell
def __(mo, df, pd):
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

    return null_counts, zero_counts, bio_features, missing_summary


@app.cell
def __(mo, missing_summary):
    mo.md(
        """
        **Finding**: No explicit nulls, but zeros in biological features represent missing data!

        - **Glucose**: Cannot be 0 (person would be dead)
        - **BloodPressure**: Cannot be 0 (no pulse)
        - **BMI**: Cannot be 0 (no body mass)
        """
    )
    return mo.ui.table(missing_summary)


@app.cell
def __(df, bio_features, plt, np):
    # Visualize missing data pattern
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Missing data heatmap
    missing_mask = df[bio_features] == 0
    axes[0].imshow(missing_mask.T, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    axes[0].set_yticks(range(len(bio_features)))
    axes[0].set_yticklabels(bio_features)
    axes[0].set_xlabel('Patient Index')
    axes[0].set_title('Missing Data Pattern (Red = Missing/Zero)')

    # Missing data percentage by feature
    missing_pct = (missing_mask.sum() / len(df) * 100).sort_values(ascending=True)
    axes[1].barh(range(len(missing_pct)), missing_pct.values)
    axes[1].set_yticks(range(len(missing_pct)))
    axes[1].set_yticklabels(missing_pct.index)
    axes[1].set_xlabel('Percentage Missing (%)')
    axes[1].set_title('Missing Data by Feature')
    axes[1].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    axes[1].legend()

    plt.tight_layout()
    missing_plot = fig
    plt.close()

    return missing_plot, missing_mask, missing_pct


@app.cell
def __(mo, missing_plot):
    mo.md("### Missing Data Visualization")
    return mo.as_html(missing_plot)


@app.cell
def __(mo, missing_pct):
    mo.md(
        f"""
        **Key Findings**:
        - **Insulin**: {missing_pct['Insulin']:.1f}% missing (nearly half!)
        - **SkinThickness**: {missing_pct['SkinThickness']:.1f}% missing
        - **BloodPressure**: {missing_pct['BloodPressure']:.1f}% missing
        - **BMI**: {missing_pct['BMI']:.1f}% missing
        - **Glucose**: {missing_pct['Glucose']:.1f}% missing (critical!)

        **Implication**: Simple deletion would lose ~50% of data.
        **Solution**: Median imputation for non-zero values (implemented in pipeline).
        """
    )
    return


@app.cell
def __(mo):
    mo.md("## 3. Univariate Analysis: Feature Distributions")
    return


@app.cell
def __(df, plt, np):
    # Create distribution plots for all features
    features = df.columns.drop('Outcome')
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Replace zeros with NaN for visualization
        data_nonzero = df[feature].replace(0, np.nan)

        # Histogram
        ax.hist(data_nonzero.dropna(), bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{feature} Distribution (zeros excluded)')

        # Add statistics
        mean_val = data_nonzero.mean()
        median_val = data_nonzero.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
        ax.legend()

    plt.tight_layout()
    dist_plot = fig
    plt.close()

    return dist_plot, features


@app.cell
def __(mo, dist_plot):
    mo.md("### Feature Distributions (Excluding Zeros)")
    return mo.as_html(dist_plot)


@app.cell
def __(df, pd, np, bio_features):
    # Statistical summary
    summary_stats = df.describe().T

    # Add skewness and kurtosis
    summary_stats['skewness'] = df.skew()
    summary_stats['kurtosis'] = df.kurtosis()

    # Add zero counts for bio features
    summary_stats['zeros'] = (df == 0).sum()
    summary_stats['zeros_pct'] = (df == 0).mean() * 100

    return summary_stats


@app.cell
def __(mo, summary_stats):
    mo.md("### Statistical Summary")
    return mo.ui.table(summary_stats.round(2))


@app.cell
def __(mo):
    mo.md(
        """
        **Observations**:
        - **Glucose** is right-skewed (expected for diabetic population)
        - **Insulin** highly variable and skewed
        - **Age** ranges 21-81, median ~29 (relatively young cohort)
        - **BMI** median ~32 (obese range, expected for at-risk population)
        """
    )
    return


@app.cell
def __(mo):
    mo.md("## 4. Bivariate Analysis: Features vs. Diabetes Outcome")
    return


@app.cell
def __(df, plt, np, features):
    # Box plots by outcome
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Replace zeros with NaN
        data_clean = df[[feature, 'Outcome']].copy()
        data_clean.loc[data_clean[feature] == 0, feature] = np.nan

        # Box plot by outcome
        data_clean.boxplot(column=feature, by='Outcome', ax=ax)
        ax.set_xlabel('Outcome (0=No Diabetes, 1=Diabetes)')
        ax.set_ylabel(feature)
        ax.set_title(f'{feature} by Diabetes Status')
        plt.sca(ax)
        plt.xticks([1, 2], ['No Diabetes', 'Diabetes'])

    plt.tight_layout()
    bivariate_plot = fig
    plt.close()

    return bivariate_plot, data_clean


@app.cell
def __(mo, bivariate_plot):
    mo.md("### Feature Distributions by Diabetes Status")
    return mo.as_html(bivariate_plot)


@app.cell
def __(df, pd, np, features):
    # Statistical comparison by outcome
    comparison = pd.DataFrame()

    for feature in features:
        data_clean = df[feature].replace(0, np.nan)

        no_diabetes = data_clean[df['Outcome'] == 0].dropna()
        has_diabetes = data_clean[df['Outcome'] == 1].dropna()

        comparison.loc[feature, 'No Diabetes (Mean)'] = no_diabetes.mean()
        comparison.loc[feature, 'Diabetes (Mean)'] = has_diabetes.mean()
        comparison.loc[feature, 'Difference'] = has_diabetes.mean() - no_diabetes.mean()
        comparison.loc[feature, 'Difference %'] = ((has_diabetes.mean() - no_diabetes.mean()) / no_diabetes.mean() * 100)

    comparison = comparison.sort_values('Difference %', ascending=False)

    return comparison, no_diabetes, has_diabetes


@app.cell
def __(mo, comparison):
    mo.md("### Mean Differences by Diabetes Status")
    return mo.ui.table(comparison.round(2))


@app.cell
def __(mo, comparison):
    mo.md(
        f"""
        **Key Discriminators** (largest differences):

        1. **{comparison.index[0]}**: +{comparison.iloc[0]['Difference %']:.1f}% higher in diabetic group
        2. **{comparison.index[1]}**: +{comparison.iloc[1]['Difference %']:.1f}% higher in diabetic group
        3. **{comparison.index[2]}**: +{comparison.iloc[2]['Difference %']:.1f}% higher in diabetic group

        These features will likely be most important for prediction.
        """
    )
    return


@app.cell
def __(mo):
    mo.md("## 5. Multivariate Analysis: Feature Correlations")
    return


@app.cell
def __(df, plt, sns, np):
    # Replace zeros with NaN for correlation
    df_clean = df.replace(0, np.nan)

    # Correlation matrix
    corr_matrix = df_clean.corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix (Zeros Excluded)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    corr_plot = fig
    plt.close()

    return df_clean, corr_matrix, corr_plot


@app.cell
def __(mo, corr_plot):
    mo.md("### Correlation Heatmap")
    return mo.as_html(corr_plot)


@app.cell
def __(corr_matrix):
    # Get correlations with Outcome
    outcome_corr = corr_matrix['Outcome'].drop('Outcome').sort_values(ascending=False)

    return outcome_corr,


@app.cell
def __(mo, outcome_corr):
    mo.md("### Correlation with Diabetes Outcome")

    corr_df = outcome_corr.reset_index()
    corr_df.columns = ['Feature', 'Correlation']

    return mo.ui.table(corr_df.round(3))


@app.cell
def __(mo, outcome_corr):
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
    return strongest_feature, strongest_corr


@app.cell
def __(mo):
    mo.md("## 6. Clinical Context Analysis")
    return


@app.cell
def __(df, plt):
    # Glucose clinical thresholds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Glucose distribution with clinical thresholds
    glucose_clean = df[df['Glucose'] > 0]['Glucose']
    axes[0].hist(glucose_clean, bins=40, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=100, color='yellow', linestyle='--', linewidth=2, label='Prediabetes (100)')
    axes[0].axvline(x=126, color='red', linestyle='--', linewidth=2, label='Diabetes (126)')
    axes[0].set_xlabel('Glucose (mg/dL)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Glucose Distribution with Clinical Thresholds')
    axes[0].legend()

    # BMI distribution with clinical categories
    bmi_clean = df[df['BMI'] > 0]['BMI']
    axes[1].hist(bmi_clean, bins=40, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=18.5, color='blue', linestyle='--', linewidth=2, label='Underweight')
    axes[1].axvline(x=25, color='green', linestyle='--', linewidth=2, label='Normal')
    axes[1].axvline(x=30, color='orange', linestyle='--', linewidth=2, label='Obese')
    axes[1].set_xlabel('BMI (kg/m²)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('BMI Distribution with Clinical Categories')
    axes[1].legend()

    plt.tight_layout()
    clinical_plot = fig
    plt.close()

    return clinical_plot, glucose_clean, bmi_clean


@app.cell
def __(mo, clinical_plot):
    mo.md("### Clinical Threshold Analysis")
    return mo.as_html(clinical_plot)


@app.cell
def __(df, np):
    # Calculate clinical category distributions
    glucose_clean = df[df['Glucose'] > 0]
    bmi_clean = df[df['BMI'] > 0]

    clinical_stats = {
        'Glucose Normal (<100)': (glucose_clean['Glucose'] < 100).sum(),
        'Glucose Prediabetes (100-125)': ((glucose_clean['Glucose'] >= 100) &
                                           (glucose_clean['Glucose'] < 126)).sum(),
        'Glucose Diabetes (≥126)': (glucose_clean['Glucose'] >= 126).sum(),
        'BMI Normal (<25)': (bmi_clean['BMI'] < 25).sum(),
        'BMI Overweight (25-30)': ((bmi_clean['BMI'] >= 25) & (bmi_clean['BMI'] < 30)).sum(),
        'BMI Obese (≥30)': (bmi_clean['BMI'] >= 30).sum(),
    }

    return clinical_stats, glucose_clean, bmi_clean


@app.cell
def __(mo, clinical_stats):
    mo.md("### Clinical Category Distribution")

    import pandas as pd
    stats_df = pd.DataFrame(list(clinical_stats.items()), columns=['Category', 'Count'])

    return mo.ui.table(stats_df)


@app.cell
def __(mo):
    mo.md("## 7. Subgroup & Fairness Analysis")
    return


@app.cell
def __(df, pd, np):
    # Create age groups
    df_analysis = df.copy()
    df_analysis['Age_Group'] = pd.cut(df_analysis['Age'],
                                       bins=[0, 30, 45, 60, 100],
                                       labels=['<30', '30-45', '45-60', '>60'])

    # Create BMI categories
    df_analysis['BMI_Category'] = pd.cut(df_analysis['BMI'].replace(0, np.nan),
                                          bins=[0, 25, 30, 100],
                                          labels=['Normal/Under', 'Overweight', 'Obese'])

    return df_analysis,


@app.cell
def __(df_analysis, pd):
    # Diabetes rate by subgroup
    age_outcome = df_analysis.groupby('Age_Group')['Outcome'].agg(['mean', 'count'])
    age_outcome.columns = ['Diabetes Rate', 'Sample Size']

    bmi_outcome = df_analysis.groupby('BMI_Category')['Outcome'].agg(['mean', 'count'])
    bmi_outcome.columns = ['Diabetes Rate', 'Sample Size']

    return age_outcome, bmi_outcome


@app.cell
def __(mo, age_outcome):
    mo.md("### Diabetes Rate by Age Group")
    return mo.ui.table(age_outcome.round(3))


@app.cell
def __(mo, bmi_outcome):
    mo.md("### Diabetes Rate by BMI Category")
    return mo.ui.table(bmi_outcome.round(3))


@app.cell
def __(df_analysis, plt):
    # Subgroup visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Age group analysis
    age_stats = df_analysis.groupby('Age_Group')['Outcome'].mean() * 100
    axes[0].bar(range(len(age_stats)), age_stats.values, color='skyblue', edgecolor='black')
    axes[0].set_xticks(range(len(age_stats)))
    axes[0].set_xticklabels(age_stats.index)
    axes[0].set_ylabel('Diabetes Rate (%)')
    axes[0].set_xlabel('Age Group')
    axes[0].set_title('Diabetes Rate by Age Group')
    axes[0].axhline(y=df_analysis['Outcome'].mean() * 100, color='red',
                     linestyle='--', label='Overall Rate')
    axes[0].legend()

    # BMI category analysis
    bmi_stats = df_analysis.groupby('BMI_Category')['Outcome'].mean() * 100
    axes[1].bar(range(len(bmi_stats)), bmi_stats.values, color='coral', edgecolor='black')
    axes[1].set_xticks(range(len(bmi_stats)))
    axes[1].set_xticklabels(bmi_stats.index, rotation=15)
    axes[1].set_ylabel('Diabetes Rate (%)')
    axes[1].set_xlabel('BMI Category')
    axes[1].set_title('Diabetes Rate by BMI Category')
    axes[1].axhline(y=df_analysis['Outcome'].mean() * 100, color='red',
                     linestyle='--', label='Overall Rate')
    axes[1].legend()

    plt.tight_layout()
    subgroup_plot = fig
    plt.close()

    return subgroup_plot, age_stats, bmi_stats


@app.cell
def __(mo, subgroup_plot):
    mo.md("### Subgroup Analysis Visualization")
    return mo.as_html(subgroup_plot)


@app.cell
def __(mo):
    mo.md(
        """
        **Fairness Observations**:
        - Diabetes rate increases with age (expected biological trend)
        - Diabetes rate increases with BMI (strong clinical association)
        - **Potential bias**: Undersampling of >60 age group
        - **Recommendation**: Monitor model performance across all subgroups
        """
    )
    return


@app.cell
def __(mo):
    mo.md("## 8. Outlier Detection")
    return


@app.cell
def __(df, np, features):
    # Z-score based outlier detection
    outlier_summary = {}

    for feature in features:
        data_clean = df[feature].replace(0, np.nan).dropna()
        z_scores = np.abs((data_clean - data_clean.mean()) / data_clean.std())
        outliers = z_scores > 3
        outlier_summary[feature] = {
            'count': outliers.sum(),
            'percentage': outliers.mean() * 100
        }

    import pandas as pd
    outlier_df = pd.DataFrame(outlier_summary).T
    outlier_df = outlier_df.sort_values('count', ascending=False)

    return outlier_summary, outlier_df, z_scores, outliers


@app.cell
def __(mo, outlier_df):
    mo.md("### Outlier Detection (Z-score > 3)")
    return mo.ui.table(outlier_df.round(2))


@app.cell
def __(mo):
    mo.md(
        """
        **Finding**: Few extreme outliers (all < 5% of data)

        **Recommendation**:
        - Keep outliers (may be clinically valid extreme cases)
        - Robust model (tree-based) will handle naturally
        - Monitor outlier influence in model evaluation
        """
    )
    return


@app.cell
def __(mo):
    mo.md("## 9. Key Findings & Recommendations")
    return


@app.cell
def __(mo):
    mo.md(
        """
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
        """
    )
    return


if __name__ == "__main__":
    app.run()
