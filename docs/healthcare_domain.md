# Healthcare Domain Knowledge Document

## 1. Clinical Informatics Foundation

### 1.1 The Health Outcomes Tetrahedron
Healthcare outcomes are governed by the four-way relationship:
- **Demographics** (age, sex, comorbidities, genetic markers)
- **Clinical Encounters** (diagnoses, procedures, medications, lab results)
- **Treatment Pathways** (drug regimens, surgical interventions, rehabilitation)
- **Health Outcomes** (survival, readmission, complication rates, QALYs)

### 1.2 Clinical Coding Systems
- **ICD-10-CM**: 72,000+ codes for diagnoses (e.g., E11.65 = T2DM with hyperglycaemia)
- **CPT**: 10,000+ procedure codes determining reimbursement
- **NDC**: 10-digit medication identifiers linked to ATC classification
- **LOINC**: 90,000+ codes for laboratory observations

### 1.3 Hospital Readmission
30-day all-cause readmission is a key quality metric tracked by CMS. The Hospital Readmissions Reduction Program (HRRP) penalises hospitals with excess readmissions for specific conditions (AMI, heart failure, pneumonia, COPD, hip/knee replacement, CABG).

Key readmission risk factors identified in this project:
- Prior inpatient admissions (strongest predictor)
- Number of emergency visits
- Medication burden (polypharmacy)
- Length of stay
- Diabetes diagnosis with poor glycaemic control

## 2. Disease Progression Models

### 2.1 SIR/SEIR Epidemiological Models
- dS/dt = −βSI/N (Susceptible → Infected at rate β)
- dI/dt = βSI/N − γI (Recovery at rate γ)
- R₀ = β/γ determines outbreak dynamics
- Financial impact: R₀ > 1 triggers insurance reserve adjustments

### 2.2 Chronic Disease Trajectories
- Diabetes: HbA1c trajectories predict complication onset
- CKD: eGFR decline curves predict dialysis timing and cost
- Heart failure: NYHA class progression drives hospitalisation frequency

## 3. Pharmacology

### 3.1 Drug Development Pipeline
| Phase | Duration | Success Rate | Purpose |
|---|---|---|---|
| Preclinical | ~5 years | 10% | Safety screening |
| Phase I | 1-2 years | 63% | Dosing/safety in healthy volunteers |
| Phase II | 2-3 years | 31% | Efficacy in 100-300 patients |
| Phase III | 3-4 years | 58% | Large-scale efficacy (1000-5000 patients) |
| Regulatory | 1-2 years | 85% | FDA/EMA review |

### 3.2 Drug-Drug Interactions
Polypharmacy (5+ concurrent medications) creates DDI risk through:
- **Pharmacokinetic**: CYP450 enzyme inhibition/induction
- **Pharmacodynamic**: Additive or antagonistic effects
- HealthRisk AI models DDI networks as graphs for risk prediction

## 4. Healthcare Data Quality

### 4.1 Challenges
- ICD-10 coding error rates: 10-30% for complex cases
- Missing data is not random (sicker patients have more complete data)
- Institutional variation in coding practices
- HIPAA constraints limit available features

### 4.2 HealthRisk AI Quality Protocol
1. Completeness audit
2. Accuracy validation
3. Consistency checks
4. Timeliness assessment
5. Conformity verification
6. Deduplication

## 5. Data Sources
- **MIMIC-IV**: 300,000+ admissions from Beth Israel Deaconess Medical Center
- **UCI Diabetes 130-US Hospitals**: 101,766 diabetic encounters (primary dataset for this project)
- **ClinicalTrials.gov**: 450,000+ clinical studies
- **FDA FAERS**: 20M+ adverse event reports
- **CMS Public Use Files**: Medicare/Medicaid claims data
