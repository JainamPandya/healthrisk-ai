# HealthRisk Lab — Game Design Document

## Overview

HealthRisk Lab is a gamified simulation platform where participants manage a **$500M diversified healthcare-financial portfolio** while responding to realistic health-sector shock scenarios. An AI opponent using HealthRisk AI models competes against the player.

## Portfolio Components

| Component | Allocation | Description |
|---|---|---|
| Health Insurance Book | $150M | 50,000 insured lives with varying risk profiles |
| Hospital Bond Portfolio | $150M | 15 hospital municipal bonds with varying credit ratings |
| Pharmaceutical Equity Portfolio | $100M | 20 pharma/biotech stocks at various pipeline stages |
| Health-Sector Credit Facility | $100M | Revolving credit to 10 healthcare systems |

## Game Modes

### Mode 1: Actuarial Challenge
- Focus: Health insurance pricing and reserve management
- Duration: 5 simulated years (20 quarters)
- Scoring: MLR compliance, reserve adequacy, profitability, member retention

### Mode 2: Hospital Credit Analyst
- Focus: Hospital bond portfolio management
- Duration: 10 simulated years (40 quarters)
- Scoring: Default avoidance, spread capture, early warning detection

### Mode 3: Pharmaceutical Portfolio Manager
- Focus: Biotech/pharma equity portfolio
- Duration: 5 simulated years (20 quarters)
- Scoring: Absolute return, Sharpe ratio, maximum drawdown

### Mode 4: Integrated Portfolio (Master Mode)
- Focus: Full $500M portfolio across all four components
- Duration: 10 simulated years (40 quarters)
- Scoring: Composite 1000-point score

## Scoring Framework (1000 Points)

| Category | Points |
|---|---|
| Insurance Management | 250 |
| Hospital Credit Analysis | 250 |
| Pharmaceutical Alpha | 200 |
| Credit Facility Management | 150 |
| Cross-Asset Risk Management | 100 |
| Speed and Decision Quality | 50 |

## Scenario Types (10 Implemented)

1. **Pandemic Outbreak** — Novel respiratory pathogen (R₀=2.5, IFR=0.8%)
2. **FDA Black Box Warning** — Safety warning for widely-prescribed medication
3. **Medicare Reimbursement Cut** — CMS announces 3% rate reduction
4. **Hospital System Merger** — Two portfolio hospital systems merge
5. **Breakthrough Drug Approval** — FDA accelerated approval for gene therapy
6. **Seasonal Flu Surge** — 40% worse than average flu season
7. **Opioid Settlement Ruling** — Major opioid settlement impacts
8. **Clinical Trial Failure** — Phase III trial fails primary endpoint
9. **Rural Hospital Closure Wave** — Multiple rural hospitals announce closure
10. **Gene Therapy Pricing Shock** — CMS coverage for $3.5M gene therapy

## AI Opponent

The AI opponent uses the full HealthRisk AI model suite to make portfolio decisions, providing a benchmark. It adjusts:
- IBNR reserves based on epidemiological signals
- Hospital bond exposure based on clinical quality deterioration
- Pharmaceutical positions based on clinical trial intelligence
- Credit facility monitoring intensity based on covenant proximity

## Educational Objectives

1. Develop cross-domain analytical skills (healthcare × finance)
2. Practice real-time decision-making under uncertainty
3. Understand how clinical intelligence drives financial outcomes
4. Learn portfolio risk management in healthcare contexts

## Implementation

The simulation engine is implemented in `simulation/healthrisk_lab.py` with:
- `HealthRiskLabEngine` — main simulation controller
- `generate_scenario()` — stochastic scenario generation
- `Portfolio` — dataclass managing all portfolio components
- Quarterly cycle advancing with impact calculation and scoring
