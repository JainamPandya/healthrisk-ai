# Feature Engineering Documentation

## Overview
This document describes the feature engineering pipeline for HealthRisk AI, covering clinical, financial, and cross-domain features.

## 1. Clinical Feature Engineering

### 1.1 Diagnosis Features (ICD-10 Engineering)
- **Raw ICD-10 codes**: `diag_1`, `diag_2`, `diag_3` contain diagnosis codes
- **Hierarchical encoding**: Chapter level (first character), Block level (first 3 characters), Full code level
- **HCC mapping**: Diagnosis codes are mapped to CMS Hierarchical Condition Categories with associated risk weights
- **Comorbidity indices**: Charlson Comorbidity Index computed from diagnosis code combinations
- **Diagnosis count**: `number_diagnoses` — total number of diagnoses per encounter

### 1.2 Medication Features
- **Medication columns**: 23 medication features (metformin, insulin, glipizide, etc.)
- **Encoding**: Categorical values (No, Steady, Up, Down) → ordinal encoding
- **Polypharmacy score**: Count of medications with non-"No" values
- **Drug change indicator**: `change` column indicates medication changes during encounter
- **Diabetes medication flag**: `diabetesMed` binary indicator

### 1.3 Laboratory Value Features
- **HbA1c result**: `A1Cresult` — Categorical (None, Norm, >7, >8)
- **Glucose serum**: `max_glu_serum` — Categorical (None, Norm, >200, >300)
- **Lab procedure count**: `num_lab_procedures` — proxy for clinical complexity

### 1.4 Utilisation Features
- **Inpatient visits**: `number_inpatient` — prior inpatient admissions (strongest predictor)
- **Emergency visits**: `number_emergency` — prior ED visits
- **Outpatient visits**: `number_outpatient` — prior outpatient encounters
- **Time in hospital**: `time_in_hospital` — length of current stay
- **Procedure count**: `num_procedures` — procedures during current encounter

### 1.5 Demographic Features
- **Age**: Categorical age buckets ([0-10), [10-20), ..., [90-100))
- **Gender**: Binary (Male/Female)
- **Race**: Categorical (Caucasian, African American, Hispanic, Asian, Other)
- **Weight**: Mostly missing, not used in primary model

## 2. Financial Feature Engineering

### 2.1 Insurance Pricing Features
- **HCC risk score**: Computed from diagnosis codes using CMS-HCC model
- **Age rating factor**: ACA 3:1 age band factor
- **Geographic factor**: Regional cost adjustment
- **Clinical risk loading**: Dollar amount per HCC risk unit

### 2.2 Hospital Credit Risk Features
- **Operating margin**: (Revenue − Expenses) / Revenue
- **DSCR**: Debt Service Coverage Ratio
- **Days cash on hand**: Liquidity measure
- **Payer mix**: Medicare / Medicaid / Commercial distribution
- **Case Mix Index**: Average DRG weight indicating complexity

### 2.3 Pharmaceutical Signal Features
- **Enrollment velocity**: Trial enrollment rate vs. target
- **Phase success probability**: Indication-adjusted probability
- **Patent runway**: Years of remaining patent protection
- **Pipeline rNPV**: Risk-adjusted Net Present Value

## 3. Cross-Domain Fusion Features

### 3.1 Clinical-Financial Linkage
- Clinical trajectory risk score → Insurance premium adjustment
- Readmission rate → Hospital bond credit quality
- Drug interaction score → Patient risk stratification tier

### 3.2 Preprocessing Pipeline
The scikit-learn `ColumnTransformer` applies:
- **Numerical features**: `SimpleImputer(strategy='median')` → `StandardScaler`
- **Categorical features**: `SimpleImputer(strategy='most_frequent')` → `OrdinalEncoder`

## 4. Feature Validation
- All features verified for logical consistency (e.g., non-negative counts)
- Missing data patterns analysed: clinical data is missing-not-at-random
- Feature importance validated against clinical domain knowledge
- Top 5 features by SHAP importance: `number_inpatient`, `number_emergency`, `num_medications`, `time_in_hospital`, `number_diagnoses`
