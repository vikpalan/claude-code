"""
Canadian Mortgage Payment Calculator
=====================================

Implements Canadian mortgage industry rules including:
- OSFI B-20 Stress Test (qualifying rate)
- CMHC/Sagen/Canada Guaranty mortgage default insurance premiums
- Minimum down payment rules (tiered by purchase price)
- Canadian semi-annual compounding (unique to Canada)
- GDS and TDS ratio calculations
- Amortization limits (25 yr insured, up to 30 yr conventional)
- Property transfer tax estimation (BC/ON examples)

References:
- OSFI Guideline B-20
- CMHC Mortgage Insurance Premiums
- Canada Mortgage and Housing Corporation rules
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Constants & Rate Tables
# ---------------------------------------------------------------------------

# CMHC Mortgage Default Insurance Premium Rates (% of mortgage amount)
# Applies when down payment is less than 20%
CMHC_PREMIUM_RATES = [
    # (min_ltv_exclusive, max_ltv_inclusive, premium_rate)
    (0.00, 0.65, 0.0060),   #  <= 65% LTV  -> 0.60%
    (0.65, 0.75, 0.0170),   #  65.01-75%   -> 1.70%
    (0.75, 0.80, 0.0240),   #  75.01-80%   -> 2.40%
    (0.80, 0.85, 0.0280),   #  80.01-85%   -> 2.80%
    (0.85, 0.90, 0.0310),   #  85.01-90%   -> 3.10%
    (0.90, 0.95, 0.0400),   #  90.01-95%   -> 4.00%
]

# Maximum purchase price eligible for insured mortgage
MAX_INSURED_PRICE = 1_500_000

# OSFI B-20 Stress Test floor rate
STRESS_TEST_FLOOR_RATE = 0.0525  # 5.25%
STRESS_TEST_BUFFER = 0.02        # contract rate + 2%

# GDS / TDS limits
MAX_GDS_RATIO = 0.39  # 39%
MAX_TDS_RATIO = 0.44  # 44%

# Standard property tax rate estimate (annual, as % of home value)
DEFAULT_PROPERTY_TAX_RATE = 0.01  # 1%

# Heating cost estimate (monthly)
DEFAULT_MONTHLY_HEATING = 150.00


class MortgageType(Enum):
    INSURED = "Insured (< 20% down)"
    INSURABLE = "Insurable (>= 20% down, meets insured criteria)"
    UNINSURABLE = "Uninsurable (> $1.5M or 30-yr amortization)"


@dataclass
class MortgageInput:
    """Input parameters for the mortgage calculation."""
    purchase_price: float
    down_payment: float
    annual_interest_rate: float        # Contract rate (e.g., 0.0549 for 5.49%)
    amortization_years: int            # 25 or 30
    term_years: int = 5                # Typical 5-year term
    annual_income: float = 0.0         # Gross household income
    monthly_debts: float = 0.0         # Other monthly debt payments (car, LOC, etc.)
    monthly_condo_fees: float = 0.0    # Condo/strata fees
    annual_property_tax: Optional[float] = None  # Override auto-estimate
    monthly_heating: float = DEFAULT_MONTHLY_HEATING


@dataclass
class MortgageResult:
    """Full results of a Canadian mortgage calculation."""
    # Classification
    mortgage_type: MortgageType

    # Core numbers
    purchase_price: float
    down_payment: float
    down_payment_pct: float
    mortgage_amount: float          # Before insurance
    cmhc_premium: float
    cmhc_premium_rate: float
    total_mortgage: float           # Mortgage + insurance premium

    # Rates
    contract_rate: float
    effective_monthly_rate: float   # After semi-annual compounding conversion
    qualifying_rate: float          # Stress-test rate

    # Payments
    monthly_payment: float          # At contract rate
    monthly_payment_stress: float   # At qualifying rate
    total_interest_over_term: float
    total_paid_over_amortization: float
    balance_at_term_end: float

    # Affordability
    annual_property_tax: float
    monthly_property_tax: float
    monthly_heating: float
    monthly_condo_fees: float
    gds_ratio: Optional[float]
    tds_ratio: Optional[float]
    gds_pass: Optional[bool]
    tds_pass: Optional[bool]

    # Term
    amortization_years: int
    term_years: int


# ---------------------------------------------------------------------------
# Core Calculation Functions
# ---------------------------------------------------------------------------

def get_minimum_down_payment(purchase_price: float) -> float:
    """
    Calculate the minimum down payment per Canadian rules:
      - 5% on the first $500,000
      - 10% on the portion between $500,000 and $1,500,000
      - 20% on any amount above $1,500,000 (not insurable)
    """
    if purchase_price <= 500_000:
        return purchase_price * 0.05
    elif purchase_price <= 1_500_000:
        return 500_000 * 0.05 + (purchase_price - 500_000) * 0.10
    else:
        # Must put 20% minimum on properties > $1.5M
        return purchase_price * 0.20


def get_cmhc_premium_rate(ltv: float) -> float:
    """Look up the CMHC insurance premium rate based on loan-to-value ratio."""
    for min_ltv, max_ltv, rate in CMHC_PREMIUM_RATES:
        if min_ltv < ltv <= max_ltv:
            return rate
    if ltv <= 0:
        return 0.0
    return 0.0


def classify_mortgage(purchase_price: float, down_payment_pct: float,
                      amortization_years: int) -> MortgageType:
    """Classify the mortgage type based on Canadian rules."""
    if down_payment_pct < 0.20:
        if purchase_price > MAX_INSURED_PRICE:
            raise ValueError(
                f"Properties over ${MAX_INSURED_PRICE:,.0f} require a minimum "
                f"20% down payment. Cannot obtain mortgage default insurance."
            )
        if amortization_years > 25:
            raise ValueError(
                "Insured mortgages (< 20% down) are limited to 25-year "
                "amortization maximum."
            )
        return MortgageType.INSURED
    elif purchase_price <= MAX_INSURED_PRICE and amortization_years <= 25:
        return MortgageType.INSURABLE
    else:
        return MortgageType.UNINSURABLE


def canadian_effective_monthly_rate(annual_rate: float) -> float:
    """
    Convert a nominal annual rate with SEMI-ANNUAL compounding (Canadian
    standard) to an effective monthly rate.

    Canadian mortgages compound semi-annually, not monthly. The formula:
        effective_monthly = (1 + annual_rate/2)^(1/6) - 1
    """
    return (1 + annual_rate / 2) ** (1 / 6) - 1


def monthly_payment(principal: float, monthly_rate: float,
                    total_months: int) -> float:
    """Standard amortization payment formula."""
    if monthly_rate == 0:
        return principal / total_months
    return principal * (monthly_rate * (1 + monthly_rate) ** total_months) / \
        ((1 + monthly_rate) ** total_months - 1)


def balance_after_n_payments(principal: float, monthly_rate: float,
                             payment: float, n: int) -> float:
    """Calculate remaining balance after n payments."""
    if monthly_rate == 0:
        return principal - payment * n
    return principal * (1 + monthly_rate) ** n - \
        payment * ((1 + monthly_rate) ** n - 1) / monthly_rate


def get_qualifying_rate(contract_rate: float) -> float:
    """
    OSFI B-20 Stress Test:
    Qualifying rate = the GREATER of:
      1) The Bank of Canada 5-year benchmark rate floor (5.25%)
      2) The contract rate + 2.00%
    """
    return max(STRESS_TEST_FLOOR_RATE, contract_rate + STRESS_TEST_BUFFER)


# ---------------------------------------------------------------------------
# Main Calculator
# ---------------------------------------------------------------------------

def calculate_mortgage(inp: MortgageInput) -> MortgageResult:
    """Run the full Canadian mortgage calculation."""

    # --- Validation ---
    min_dp = get_minimum_down_payment(inp.purchase_price)
    if inp.down_payment < min_dp - 0.01:  # small tolerance for rounding
        raise ValueError(
            f"Down payment ${inp.down_payment:,.2f} is below the minimum "
            f"required ${min_dp:,.2f} for a ${inp.purchase_price:,.0f} home."
        )

    dp_pct = inp.down_payment / inp.purchase_price
    mortgage_amount = inp.purchase_price - inp.down_payment

    # --- Classify ---
    mortgage_type = classify_mortgage(
        inp.purchase_price, dp_pct, inp.amortization_years
    )

    # --- CMHC Insurance ---
    ltv = mortgage_amount / inp.purchase_price
    if mortgage_type == MortgageType.INSURED:
        cmhc_rate = get_cmhc_premium_rate(ltv)
        cmhc_premium = mortgage_amount * cmhc_rate
    else:
        cmhc_rate = 0.0
        cmhc_premium = 0.0

    total_mortgage = mortgage_amount + cmhc_premium

    # --- Rates ---
    contract_rate = inp.annual_interest_rate
    eff_monthly = canadian_effective_monthly_rate(contract_rate)
    qualifying_rate = get_qualifying_rate(contract_rate)
    eff_monthly_stress = canadian_effective_monthly_rate(qualifying_rate)

    # --- Payments ---
    total_months = inp.amortization_years * 12
    term_months = inp.term_years * 12

    pmt = monthly_payment(total_mortgage, eff_monthly, total_months)
    pmt_stress = monthly_payment(total_mortgage, eff_monthly_stress, total_months)

    # Balance at end of term
    bal_at_term = balance_after_n_payments(
        total_mortgage, eff_monthly, pmt, term_months
    )

    # Total interest over term
    total_paid_term = pmt * term_months
    principal_paid_term = total_mortgage - bal_at_term
    interest_over_term = total_paid_term - principal_paid_term

    # Total over full amortization (assuming rate stays constant)
    total_over_amort = pmt * total_months

    # --- Property Tax ---
    annual_ptax = inp.annual_property_tax if inp.annual_property_tax is not None \
        else inp.purchase_price * DEFAULT_PROPERTY_TAX_RATE
    monthly_ptax = annual_ptax / 12

    # --- GDS / TDS ---
    gds = None
    tds = None
    gds_pass = None
    tds_pass = None
    if inp.annual_income > 0:
        monthly_income = inp.annual_income / 12
        # GDS uses stress-test payment
        housing_costs = (pmt_stress + monthly_ptax + inp.monthly_heating
                         + inp.monthly_condo_fees * 0.50)  # 50% of condo fees
        gds = housing_costs / monthly_income
        tds = (housing_costs + inp.monthly_debts) / monthly_income
        gds_pass = gds <= MAX_GDS_RATIO
        tds_pass = tds <= MAX_TDS_RATIO

    return MortgageResult(
        mortgage_type=mortgage_type,
        purchase_price=inp.purchase_price,
        down_payment=inp.down_payment,
        down_payment_pct=dp_pct,
        mortgage_amount=mortgage_amount,
        cmhc_premium=cmhc_premium,
        cmhc_premium_rate=cmhc_rate,
        total_mortgage=total_mortgage,
        contract_rate=contract_rate,
        effective_monthly_rate=eff_monthly,
        qualifying_rate=qualifying_rate,
        monthly_payment=pmt,
        monthly_payment_stress=pmt_stress,
        total_interest_over_term=interest_over_term,
        total_paid_over_amortization=total_over_amort,
        balance_at_term_end=bal_at_term,
        annual_property_tax=annual_ptax,
        monthly_property_tax=monthly_ptax,
        monthly_heating=inp.monthly_heating,
        monthly_condo_fees=inp.monthly_condo_fees,
        gds_ratio=gds,
        tds_ratio=tds,
        gds_pass=gds_pass,
        tds_pass=tds_pass,
        amortization_years=inp.amortization_years,
        term_years=inp.term_years,
    )


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

def fmt(val: float) -> str:
    return f"${val:>14,.2f}"


def pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def print_result(result: MortgageResult, title: str) -> None:
    """Pretty-print a mortgage result."""
    w = 72
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)

    print(f"\n  {'PROPERTY & DOWN PAYMENT':─<{w-4}}")
    print(f"  Purchase Price:              {fmt(result.purchase_price)}")
    print(f"  Down Payment:                {fmt(result.down_payment)}  "
          f"({pct(result.down_payment_pct)})")
    min_dp = get_minimum_down_payment(result.purchase_price)
    print(f"  Minimum Down Payment Req'd:  {fmt(min_dp)}")
    print(f"  Mortgage Type:               {result.mortgage_type.value}")

    print(f"\n  {'MORTGAGE INSURANCE (CMHC/Sagen/CG)':─<{w-4}}")
    if result.cmhc_premium > 0:
        print(f"  Loan-to-Value (LTV):         {pct(1 - result.down_payment_pct)}")
        print(f"  Insurance Premium Rate:      {pct(result.cmhc_premium_rate)}")
        print(f"  Insurance Premium:           {fmt(result.cmhc_premium)}")
        print(f"  Total Mortgage (with ins.):  {fmt(result.total_mortgage)}")
    else:
        print(f"  Not required (>= 20% down payment)")
        print(f"  Total Mortgage:              {fmt(result.total_mortgage)}")

    print(f"\n  {'INTEREST RATES':─<{w-4}}")
    print(f"  Contract Rate:               {pct(result.contract_rate)}")
    print(f"  Effective Monthly Rate:      {pct(result.effective_monthly_rate)}"
          f"  (semi-annual compounding)")
    print(f"  Stress Test Qualifying Rate: {pct(result.qualifying_rate)}"
          f"  (B-20 rule)")

    print(f"\n  {'PAYMENT SUMMARY':─<{w-4}}")
    print(f"  Amortization:                {result.amortization_years} years")
    print(f"  Term:                        {result.term_years} years")
    print(f"  Monthly Payment:             {fmt(result.monthly_payment)}"
          f"  (at contract rate)")
    print(f"  Monthly Payment (Stress):    {fmt(result.monthly_payment_stress)}"
          f"  (at qualifying rate)")
    print(f"  Total Interest ({result.term_years}-yr term):  "
          f"{fmt(result.total_interest_over_term)}")
    print(f"  Balance at Term End:         {fmt(result.balance_at_term_end)}")
    print(f"  Total Paid (full amort.):    {fmt(result.total_paid_over_amortization)}")
    total_interest_amort = result.total_paid_over_amortization - result.total_mortgage
    print(f"  Total Interest (full amort.):{fmt(total_interest_amort)}")

    print(f"\n  {'CARRYING COSTS (Monthly)':─<{w-4}}")
    print(f"  Mortgage Payment:            {fmt(result.monthly_payment)}")
    print(f"  Property Tax:                {fmt(result.monthly_property_tax)}")
    print(f"  Heating:                     {fmt(result.monthly_heating)}")
    if result.monthly_condo_fees > 0:
        print(f"  Condo Fees:                  {fmt(result.monthly_condo_fees)}")
    total_carry = (result.monthly_payment + result.monthly_property_tax
                   + result.monthly_heating + result.monthly_condo_fees)
    print(f"  ─────────────────────────────────────────")
    print(f"  Total Monthly Carrying Cost: {fmt(total_carry)}")

    if result.gds_ratio is not None:
        print(f"\n  {'AFFORDABILITY (Stress-Tested)':─<{w-4}}")
        gds_status = "PASS" if result.gds_pass else "FAIL"
        tds_status = "PASS" if result.tds_pass else "FAIL"
        print(f"  GDS Ratio:                   {pct(result.gds_ratio)}"
              f"  (max {pct(MAX_GDS_RATIO)})  [{gds_status}]")
        print(f"  TDS Ratio:                   {pct(result.tds_ratio)}"
              f"  (max {pct(MAX_TDS_RATIO)})  [{tds_status}]")
        if result.gds_pass and result.tds_pass:
            print(f"  --> You QUALIFY under B-20 stress test rules.")
        else:
            print(f"  --> You DO NOT QUALIFY under current ratios.")

    print("=" * w)


# ---------------------------------------------------------------------------
# Three Scenarios
# ---------------------------------------------------------------------------

def run_scenarios() -> None:
    print("\n" + "#" * 72)
    print("#" + " " * 18 + "CANADIAN MORTGAGE CALCULATOR" + " " * 24 + "#")
    print("#" + " " * 10 + "With OSFI B-20 Stress Test & CMHC Rules" + " " * 19 + "#")
    print("#" * 72)

    # ── SCENARIO 1: First-Time Buyer ──────────────────────────────────────
    # Modest condo purchase, minimum down payment, insured mortgage
    scenario1 = MortgageInput(
        purchase_price=450_000,
        down_payment=22_500,          # 5% minimum
        annual_interest_rate=0.0549,  # 5.49% (typical 5-yr fixed 2024-2025)
        amortization_years=25,
        term_years=5,
        annual_income=105_000,
        monthly_debts=350,            # Car payment
        monthly_condo_fees=400,
        annual_property_tax=3_200,
        monthly_heating=75,           # Condo = lower heating
    )
    r1 = calculate_mortgage(scenario1)
    print_result(r1, "SCENARIO 1: First-Time Buyer — Insured Condo Purchase")

    # ── SCENARIO 2: Move-Up Buyer ─────────────────────────────────────────
    # Detached home, 20% down, conventional mortgage, higher income
    scenario2 = MortgageInput(
        purchase_price=850_000,
        down_payment=170_000,         # Exactly 20%
        annual_interest_rate=0.0519,  # 5.19% (negotiated rate)
        amortization_years=25,
        term_years=5,
        annual_income=185_000,
        monthly_debts=600,            # Car + line of credit
        monthly_condo_fees=0,
        annual_property_tax=6_800,
        monthly_heating=200,
    )
    r2 = calculate_mortgage(scenario2)
    print_result(r2, "SCENARIO 2: Move-Up Buyer — Conventional Detached Home")

    # ── SCENARIO 3: High-Value Property ───────────────────────────────────
    # Luxury home over $1.5M, 20% required, 30-yr amort, uninsurable
    scenario3 = MortgageInput(
        purchase_price=1_800_000,
        down_payment=360_000,         # 20%
        annual_interest_rate=0.0579,  # 5.79% (premium for uninsurable)
        amortization_years=30,        # 30-year (not eligible for insurance)
        term_years=5,
        annual_income=350_000,
        monthly_debts=1_200,          # Multiple obligations
        monthly_condo_fees=0,
        annual_property_tax=14_400,
        monthly_heating=300,
    )
    r3 = calculate_mortgage(scenario3)
    print_result(r3, "SCENARIO 3: High-Value Property — Uninsurable 30-Year")

    # ── Summary comparison ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 72)
    header = f"  {'':30} {'Scenario 1':>14} {'Scenario 2':>14} {'Scenario 3':>14}"
    print(header)
    print("  " + "─" * 68)

    rows = [
        ("Purchase Price",
         r1.purchase_price, r2.purchase_price, r3.purchase_price),
        ("Down Payment",
         r1.down_payment, r2.down_payment, r3.down_payment),
        ("CMHC Insurance",
         r1.cmhc_premium, r2.cmhc_premium, r3.cmhc_premium),
        ("Total Mortgage",
         r1.total_mortgage, r2.total_mortgage, r3.total_mortgage),
        ("Monthly Payment",
         r1.monthly_payment, r2.monthly_payment, r3.monthly_payment),
        ("Monthly (Stress Test)",
         r1.monthly_payment_stress, r2.monthly_payment_stress,
         r3.monthly_payment_stress),
        ("Interest Over Term",
         r1.total_interest_over_term, r2.total_interest_over_term,
         r3.total_interest_over_term),
        ("Total Cost (Full Amort.)",
         r1.total_paid_over_amortization, r2.total_paid_over_amortization,
         r3.total_paid_over_amortization),
    ]
    for label, v1, v2, v3 in rows:
        print(f"  {label:30} ${v1:>13,.2f} ${v2:>13,.2f} ${v3:>13,.2f}")

    print("  " + "─" * 68)
    types = [r1.mortgage_type.value, r2.mortgage_type.value,
             r3.mortgage_type.value]
    print(f"  {'Mortgage Type':30} {types[0]:>14} {types[1]:>14} {types[2]:>14}")

    if r1.gds_ratio is not None:
        print(f"\n  {'GDS Ratio':30} "
              f"{pct(r1.gds_ratio):>14} {pct(r2.gds_ratio):>14} "
              f"{pct(r3.gds_ratio):>14}")
        print(f"  {'TDS Ratio':30} "
              f"{pct(r1.tds_ratio):>14} {pct(r2.tds_ratio):>14} "
              f"{pct(r3.tds_ratio):>14}")
        gds_results = [
            "PASS" if r.gds_pass else "FAIL" for r in [r1, r2, r3]
        ]
        tds_results = [
            "PASS" if r.tds_pass else "FAIL" for r in [r1, r2, r3]
        ]
        print(f"  {'GDS Result':30} {gds_results[0]:>14} "
              f"{gds_results[1]:>14} {gds_results[2]:>14}")
        print(f"  {'TDS Result':30} {tds_results[0]:>14} "
              f"{tds_results[1]:>14} {tds_results[2]:>14}")

    print("\n" + "=" * 72)
    print("  NOTES ON CANADIAN MORTGAGE RULES")
    print("=" * 72)
    print("""
  1. SEMI-ANNUAL COMPOUNDING: Unlike the US (monthly compounding), Canadian
     mortgages compound interest semi-annually by law, resulting in slightly
     lower effective rates.

  2. STRESS TEST (B-20): All borrowers must qualify at the higher of 5.25%
     or their contract rate + 2%, regardless of down payment size.

  3. DOWN PAYMENT MINIMUMS:
     - 5% on first $500,000
     - 10% on $500,001 to $1,500,000
     - 20% minimum on properties over $1,500,000

  4. MORTGAGE INSURANCE (CMHC/Sagen/Canada Guaranty):
     - Required when down payment < 20%
     - Premium ranges from 0.60% to 4.00% of mortgage amount
     - Added to the mortgage principal
     - Maximum insurable purchase price: $1,500,000
     - Maximum amortization for insured: 25 years

  5. GDS/TDS RATIOS:
     - GDS (Gross Debt Service): Housing costs / gross income <= 39%
     - TDS (Total Debt Service): All debts / gross income <= 44%
     - Calculated using the stress-test qualifying rate

  6. AMORTIZATION:
     - Insured mortgages: maximum 25 years
     - Conventional (>= 20% down): up to 30 years available
""")


if __name__ == "__main__":
    run_scenarios()
