"""
HealthRisk Lab — Gamified Simulation Engine.

A portfolio simulation platform where players manage a $500M
diversified healthcare portfolio (insurance, hospital bonds,
pharmaceutical equities, credit facilities) while responding
to health-related financial shocks.

References:
- PDF Section B1: HealthRisk Lab Simulation Overview
- PDF Section B3: Scenario Engine Design
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Portfolio Components
# ---------------------------------------------------------------------------

@dataclass
class InsuranceBook:
    """Health insurance book of business."""
    value: float = 150_000_000
    members: int = 50_000
    mlr: float = 0.82
    ibnr_reserve: float = 15_000_000
    premium_per_member: float = 3_000.0
    claims_trend: float = 0.03  # annual claims cost trend


@dataclass
class HospitalBond:
    """A hospital municipal bond in the portfolio."""
    name: str
    face_value: float
    coupon_rate: float
    credit_rating: str
    maturity_years: int
    readmission_rate: float = 0.155
    occupancy: float = 0.72
    current_spread: float = 0.015  # basis points over risk-free


@dataclass
class PharmaStock:
    """A pharmaceutical equity position."""
    ticker: str
    shares: int
    price: float
    pipeline_phase: str  # phase_1, phase_2, phase_3, marketed
    indication: str
    trial_result_pending: bool = False


@dataclass
class CreditFacility:
    """Healthcare system revolving credit facility."""
    borrower: str
    commitment: float
    drawn_amount: float
    interest_rate: float
    covenant_dscr_min: float = 1.25
    current_dscr: float = 2.0


@dataclass
class Portfolio:
    """Player's $500M healthcare-financial portfolio."""
    insurance: InsuranceBook = field(default_factory=InsuranceBook)
    bonds: List[HospitalBond] = field(default_factory=list)
    stocks: List[PharmaStock] = field(default_factory=list)
    credit_facilities: List[CreditFacility] = field(default_factory=list)
    cash: float = 0.0
    score: int = 500  # Starting score out of 1000


# ---------------------------------------------------------------------------
# Scenario Engine
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A health-financial shock event."""
    name: str
    description: str
    scenario_type: str  # pandemic, drug_safety, regulatory, merger, etc.
    severity: str       # low, medium, high, critical
    insurance_impact: float   # multiplier on claims
    bond_spread_impact: float # bps change to hospital bond spreads
    pharma_impact: Dict[str, float] = field(default_factory=dict)  # ticker → price change %
    credit_draw_increase: float = 0.0  # additional draw on credit lines


SCENARIO_LIBRARY: List[Dict] = [
    {
        "name": "Novel Respiratory Pathogen Outbreak",
        "description": "R₀=2.5 respiratory virus detected. WHO monitoring closely.",
        "scenario_type": "pandemic",
        "severity": "high",
        "insurance_impact": 1.25,
        "bond_spread_impact": 0.005,
        "credit_draw_increase": 0.15,
    },
    {
        "name": "FDA Black Box Warning",
        "description": "FDA issues black box warning for a widely-prescribed medication.",
        "scenario_type": "drug_safety",
        "severity": "high",
        "insurance_impact": 1.08,
        "bond_spread_impact": 0.002,
        "credit_draw_increase": 0.05,
    },
    {
        "name": "Medicare Reimbursement Cut",
        "description": "CMS announces 3% reduction in Medicare inpatient rates.",
        "scenario_type": "regulatory",
        "severity": "medium",
        "insurance_impact": 0.98,
        "bond_spread_impact": 0.008,
        "credit_draw_increase": 0.10,
    },
    {
        "name": "Hospital System Merger",
        "description": "Two major hospital systems announce merger.",
        "scenario_type": "merger",
        "severity": "medium",
        "insurance_impact": 1.02,
        "bond_spread_impact": -0.003,
        "credit_draw_increase": 0.0,
    },
    {
        "name": "Breakthrough Drug Approval",
        "description": "FDA grants accelerated approval to a gene therapy.",
        "scenario_type": "approval",
        "severity": "low",
        "insurance_impact": 1.05,
        "bond_spread_impact": 0.0,
        "credit_draw_increase": 0.0,
    },
    {
        "name": "Seasonal Flu Surge",
        "description": "Flu season 40% worse than average. Hospital capacity strained.",
        "scenario_type": "epidemic",
        "severity": "medium",
        "insurance_impact": 1.12,
        "bond_spread_impact": 0.003,
        "credit_draw_increase": 0.08,
    },
    {
        "name": "Opioid Settlement Ruling",
        "description": "Major opioid settlement impacts pharma and insurance costs.",
        "scenario_type": "litigation",
        "severity": "high",
        "insurance_impact": 1.06,
        "bond_spread_impact": 0.001,
        "credit_draw_increase": 0.02,
    },
    {
        "name": "Clinical Trial Failure",
        "description": "Phase III trial for a major drug candidate fails primary endpoint.",
        "scenario_type": "trial_failure",
        "severity": "high",
        "insurance_impact": 1.0,
        "bond_spread_impact": 0.0,
        "credit_draw_increase": 0.0,
    },
    {
        "name": "Rural Hospital Closure Wave",
        "description": "Three rural hospitals in portfolio announce closure plans.",
        "scenario_type": "closure",
        "severity": "critical",
        "insurance_impact": 1.04,
        "bond_spread_impact": 0.015,
        "credit_draw_increase": 0.20,
    },
    {
        "name": "Gene Therapy Pricing Shock",
        "description": "CMS announces coverage determination for $3.5M gene therapy.",
        "scenario_type": "pricing",
        "severity": "medium",
        "insurance_impact": 1.15,
        "bond_spread_impact": 0.0,
        "credit_draw_increase": 0.0,
    },
]


def generate_scenario(quarter: int, seed: Optional[int] = None) -> Scenario:
    """
    Generate a health-financial scenario for a given quarter.

    Parameters
    ----------
    quarter : int
        Current simulation quarter (1-40).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    Scenario
        The generated scenario event.
    """
    rng = random.Random(seed if seed is not None else quarter)
    template = rng.choice(SCENARIO_LIBRARY)

    return Scenario(
        name=template["name"],
        description=template["description"],
        scenario_type=template["scenario_type"],
        severity=template["severity"],
        insurance_impact=template["insurance_impact"],
        bond_spread_impact=template["bond_spread_impact"],
        pharma_impact=template.get("pharma_impact", {}),
        credit_draw_increase=template.get("credit_draw_increase", 0.0),
    )


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class HealthRiskLabEngine:
    """
    Main simulation engine for HealthRisk Lab.

    Runs quarterly cycles over a configurable time horizon.
    """

    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        total_quarters: int = 40,
        seed: int = 42,
    ):
        self.portfolio = portfolio or self._create_default_portfolio()
        self.total_quarters = total_quarters
        self.seed = seed
        self.current_quarter = 0
        self.history: List[Dict] = []

    def _create_default_portfolio(self) -> Portfolio:
        """Create the default $500M portfolio."""
        bonds = [
            HospitalBond(f"Hospital Bond {i+1}", 10_000_000, 0.04 + i * 0.002,
                         ["AA", "A", "BBB+", "BBB", "BBB-"][i % 5],
                         10 + i, 0.14 + i * 0.005, 0.70 + i * 0.02)
            for i in range(15)
        ]
        stocks = [
            PharmaStock(f"PHARMA{i+1}", 10000, 50.0 + i * 10,
                        ["phase_2", "phase_3", "marketed"][i % 3],
                        ["oncology", "cardiovascular", "cns", "rare_disease"][i % 4])
            for i in range(20)
        ]
        facilities = [
            CreditFacility(f"Health System {i+1}", 10_000_000, 3_000_000,
                           0.05 + i * 0.003, 1.25, 2.0 + i * 0.1)
            for i in range(10)
        ]

        return Portfolio(
            insurance=InsuranceBook(),
            bonds=bonds,
            stocks=stocks,
            credit_facilities=facilities,
        )

    def advance_quarter(self) -> Dict:
        """
        Advance the simulation by one quarter.

        Returns
        -------
        dict
            Quarter summary including scenario, impacts, and updated portfolio state.
        """
        self.current_quarter += 1
        scenario = generate_scenario(self.current_quarter, self.seed + self.current_quarter)

        # Apply impacts
        impacts = self._apply_scenario(scenario)

        # Calculate quarterly score
        score_delta = self._calculate_score_delta(scenario, impacts)
        self.portfolio.score = max(0, min(1000, self.portfolio.score + score_delta))

        summary = {
            "quarter": self.current_quarter,
            "scenario": scenario.name,
            "scenario_type": scenario.scenario_type,
            "severity": scenario.severity,
            "impacts": impacts,
            "score_delta": score_delta,
            "total_score": self.portfolio.score,
            "portfolio_value": self._calculate_portfolio_value(),
        }
        self.history.append(summary)
        return summary

    def _apply_scenario(self, scenario: Scenario) -> Dict[str, float]:
        """Apply scenario impacts to the portfolio."""
        impacts = {}

        # Insurance impact
        quarterly_claims = (
            self.portfolio.insurance.premium_per_member
            * self.portfolio.insurance.members
            * self.portfolio.insurance.mlr
            / 4
        )
        claim_change = quarterly_claims * (scenario.insurance_impact - 1.0)
        self.portfolio.insurance.mlr = min(
            1.0, self.portfolio.insurance.mlr * scenario.insurance_impact
        )
        impacts["insurance_claim_change"] = round(claim_change, 2)

        # Bond impact
        total_spread_impact = 0.0
        for bond in self.portfolio.bonds:
            bond.current_spread += scenario.bond_spread_impact
            total_spread_impact += bond.face_value * scenario.bond_spread_impact
        impacts["bond_spread_impact"] = round(total_spread_impact, 2)

        # Credit facility impact
        total_draw_increase = 0.0
        for facility in self.portfolio.credit_facilities:
            increase = (facility.commitment - facility.drawn_amount) * scenario.credit_draw_increase
            facility.drawn_amount = min(facility.commitment, facility.drawn_amount + increase)
            total_draw_increase += increase
        impacts["credit_draw_increase"] = round(total_draw_increase, 2)

        return impacts

    def _calculate_score_delta(self, scenario: Scenario, impacts: Dict) -> int:
        """Calculate score change based on portfolio performance."""
        delta = 0

        # Insurance: penalise MLR deviation from optimal (80-85%)
        mlr = self.portfolio.insurance.mlr
        if 0.80 <= mlr <= 0.85:
            delta += 5
        elif mlr > 0.90:
            delta -= 10
        elif mlr < 0.75:
            delta -= 5

        # Bond: penalise high spread widening
        if impacts.get("bond_spread_impact", 0) > 500_000:
            delta -= 8
        elif impacts.get("bond_spread_impact", 0) < 0:
            delta += 3

        return delta

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio market value."""
        insurance_val = self.portfolio.insurance.value
        bond_val = sum(b.face_value for b in self.portfolio.bonds)
        stock_val = sum(s.shares * s.price for s in self.portfolio.stocks)
        credit_val = sum(
            f.commitment - f.drawn_amount for f in self.portfolio.credit_facilities
        )
        return insurance_val + bond_val + stock_val + credit_val + self.portfolio.cash

    def run_full_simulation(self) -> List[Dict]:
        """Run the complete simulation and return all quarter summaries."""
        results = []
        for _ in range(self.total_quarters - self.current_quarter):
            results.append(self.advance_quarter())
        return results

    def get_final_score(self) -> Dict:
        """Return the final simulation score breakdown."""
        return {
            "total_score": self.portfolio.score,
            "quarters_played": self.current_quarter,
            "final_portfolio_value": self._calculate_portfolio_value(),
            "final_mlr": round(self.portfolio.insurance.mlr, 4),
            "scenarios_encountered": len(self.history),
        }
