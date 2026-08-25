# Financial Application Document

## 1. Health Insurance Actuarial Analytics

### 1.1 Premium Pricing Framework
```
Premium = Expected Claims Cost + Risk Margin + Administrative Loading + Profit Margin
Expected Claims Cost = Claim Frequency × Claim Severity
```

HealthRisk AI enhances traditional GLM actuarial pricing by incorporating:
- HCC risk scores derived from ICD-10 diagnosis codes
- Clinical trajectory indicators (lab value trends, medication changes)
- Readmission risk predictions as additional rating factors

### 1.2 Medical Loss Ratio (MLR)
```
MLR = (Incurred Claims + Quality Improvement) / Earned Premiums
```
- ACA requires MLR ≥ 80% (individual/small group) or ≥ 85% (large group)
- HealthRisk AI monitors MLR in real-time during simulation scenarios

### 1.3 IBNR Reserve Estimation
- **Chain Ladder**: Development factors from claims triangle
- **Bornhuetter-Ferguson**: Blends chain ladder with prior expected losses
- HealthRisk AI reduces IBNR uncertainty through predictive claim emergence modelling

### 1.4 Member Risk Stratification
| Tier | Risk Score | Care Program |
|---|---|---|
| Tier 4: Catastrophic | ≥ 2.0 | Intensive Case Management |
| Tier 3: High | ≥ 1.0 | Disease Management Program |
| Tier 2: Moderate | ≥ 0.5 | Health Coaching |
| Tier 1: Low | < 0.5 | Preventive Wellness |

## 2. Hospital Credit Risk Analytics

### 2.1 Credit Risk Framework
```
Expected Loss = PD × LGD × EAD
```
- **PD**: Probability of Default from composite credit scorecard
- **LGD**: Loss Given Default (40% assumption for hospital bonds)
- **EAD**: Exposure at Default

### 2.2 Credit Scorecard Components

**Financial Metrics (60% weight)**:
- Operating margin, DSCR, liquidity, payer mix, occupancy, scale

**Clinical Quality Metrics (40% weight)**:
- 30-day readmission rate, HCAHPS stars, HAI SIR, ED boarding

### 2.3 Early Warning System
Clinical signals that predict financial deterioration 6-12 months early:
- Rising readmission rates → CMS penalty exposure
- Declining patient satisfaction → Volume loss risk
- Increasing HAI rates → Litigation risk
- Declining surgical volume → Physician departure signal
- ED boarding > 6 hours → Capacity crisis

### 2.4 Credit Rating Mapping
| Composite Score | Implied Rating |
|---|---|
| ≥ 85 | AA |
| ≥ 75 | A |
| ≥ 65 | BBB+ |
| ≥ 55 | BBB |
| ≥ 45 | BBB- |
| ≥ 35 | BB+ |
| < 35 | BB or below |

## 3. Pharmaceutical Equity Analytics

### 3.1 rNPV Valuation
```
rNPV = Σ [P(success at stage i) × CF_i / (1+r)^t_i]
```
- Phase success probabilities are indication-adjusted
- Discount rate: 8-12% for pharma
- Cash flow model: ramp-up → peak → decline lifecycle

### 3.2 Portfolio Optimisation
Mean-variance optimisation with clinical signal alpha:
- Tangency portfolio (maximum Sharpe ratio)
- Position size limits (15% max) for diversification
- Binary event risk management for Phase III readouts

### 3.3 Clinical Trial Signals
| Signal | Indicator | Investment Implication |
|---|---|---|
| Enrollment velocity | Above/below target | Timeline confidence |
| Interim analysis | DSMB recommendation | Efficacy preview |
| Competitor results | Same indication data | Market size impact |
| Adverse events | Safety signal detection | Risk assessment |
| Patent runway | Years remaining | Patent cliff exposure |

## 4. Health-Sector ESG Analytics

### 4.1 ESG Dimensions
- **Social**: Drug pricing, clinical trial diversity, opioid liability
- **Environmental**: Manufacturing impact, medical waste, anaesthetic gas emissions
- **Governance**: FDA compliance track record, board independence, data transparency

### 4.2 ESG Financial Materiality
- Drug pricing exposure: % revenue subject to IRA negotiation
- Opioid litigation reserve adequacy ratio
- Clinical trial diversity compliance risk
- Environmental remediation liability

## 5. Implementation
All financial modules are implemented in `financial/`:
- `financial/insurance/actuarial.py` — Premium pricing, IBNR, MLR, risk stratification
- `financial/credit_risk/hospital_credit.py` — Credit scorecard, PD, early warning
- `financial/pharma/pharma_analytics.py` — rNPV, pipeline valuation, portfolio optimisation
