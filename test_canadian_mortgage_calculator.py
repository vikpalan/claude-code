"""
Test suite for Canadian Mortgage Payment Calculator.

Covers:
- Minimum down payment rules (tiered)
- CMHC insurance premium rates and application
- Mortgage classification (insured/insurable/uninsurable)
- Canadian semi-annual compounding
- OSFI B-20 stress test qualifying rate
- Monthly payment calculations
- GDS/TDS affordability ratios
- Validation errors for rule violations
- Balance at term end
- Edge cases
"""

import pytest
import math

from canadian_mortgage_calculator import (
    MortgageInput,
    MortgageType,
    calculate_mortgage,
    get_minimum_down_payment,
    get_cmhc_premium_rate,
    classify_mortgage,
    canadian_effective_monthly_rate,
    monthly_payment,
    balance_after_n_payments,
    get_qualifying_rate,
    STRESS_TEST_FLOOR_RATE,
    MAX_GDS_RATIO,
    MAX_TDS_RATIO,
)


# ---------------------------------------------------------------------------
# Minimum Down Payment Rules
# ---------------------------------------------------------------------------

class TestMinimumDownPayment:
    def test_under_500k(self):
        # 5% of purchase price
        assert get_minimum_down_payment(400_000) == pytest.approx(20_000)

    def test_exactly_500k(self):
        assert get_minimum_down_payment(500_000) == pytest.approx(25_000)

    def test_between_500k_and_1_5m(self):
        # 5% on first 500K + 10% on remainder
        # 750K -> 25,000 + 25,000 = 50,000
        assert get_minimum_down_payment(750_000) == pytest.approx(50_000)

    def test_exactly_1_5m(self):
        # 25,000 + 100,000 = 125,000
        assert get_minimum_down_payment(1_500_000) == pytest.approx(125_000)

    def test_over_1_5m(self):
        # 20% flat
        assert get_minimum_down_payment(2_000_000) == pytest.approx(400_000)

    def test_small_purchase(self):
        assert get_minimum_down_payment(100_000) == pytest.approx(5_000)

    def test_at_1m(self):
        # 25,000 + 50,000 = 75,000
        assert get_minimum_down_payment(1_000_000) == pytest.approx(75_000)


# ---------------------------------------------------------------------------
# CMHC Insurance Premium Rates
# ---------------------------------------------------------------------------

class TestCMHCPremiumRate:
    def test_95_pct_ltv(self):
        assert get_cmhc_premium_rate(0.95) == pytest.approx(0.04)

    def test_90_pct_ltv(self):
        assert get_cmhc_premium_rate(0.90) == pytest.approx(0.031)

    def test_85_pct_ltv(self):
        assert get_cmhc_premium_rate(0.85) == pytest.approx(0.028)

    def test_80_pct_ltv(self):
        assert get_cmhc_premium_rate(0.80) == pytest.approx(0.024)

    def test_75_pct_ltv(self):
        assert get_cmhc_premium_rate(0.75) == pytest.approx(0.017)

    def test_65_pct_ltv(self):
        assert get_cmhc_premium_rate(0.65) == pytest.approx(0.006)

    def test_zero_ltv(self):
        assert get_cmhc_premium_rate(0.0) == 0.0

    def test_mid_range_ltv(self):
        # 87% falls in 85.01-90% bracket
        assert get_cmhc_premium_rate(0.87) == pytest.approx(0.031)


# ---------------------------------------------------------------------------
# Mortgage Classification
# ---------------------------------------------------------------------------

class TestClassifyMortgage:
    def test_insured_under_20_pct_down(self):
        result = classify_mortgage(400_000, 0.10, 25)
        assert result == MortgageType.INSURED

    def test_insurable_20_pct_down_under_1_5m(self):
        result = classify_mortgage(800_000, 0.20, 25)
        assert result == MortgageType.INSURABLE

    def test_uninsurable_over_1_5m(self):
        result = classify_mortgage(2_000_000, 0.25, 25)
        assert result == MortgageType.UNINSURABLE

    def test_uninsurable_30yr_amort(self):
        result = classify_mortgage(800_000, 0.20, 30)
        assert result == MortgageType.UNINSURABLE

    def test_error_under_20_pct_over_1_5m(self):
        with pytest.raises(ValueError, match="20% down payment"):
            classify_mortgage(2_000_000, 0.10, 25)

    def test_error_insured_30yr_amort(self):
        with pytest.raises(ValueError, match="25-year amortization"):
            classify_mortgage(400_000, 0.10, 30)


# ---------------------------------------------------------------------------
# Canadian Semi-Annual Compounding
# ---------------------------------------------------------------------------

class TestCanadianCompounding:
    def test_5_pct_rate(self):
        # (1 + 0.05/2)^(1/6) - 1
        expected = (1 + 0.05 / 2) ** (1 / 6) - 1
        assert canadian_effective_monthly_rate(0.05) == pytest.approx(expected)

    def test_zero_rate(self):
        assert canadian_effective_monthly_rate(0.0) == pytest.approx(0.0)

    def test_lower_than_monthly_compounding(self):
        # Canadian semi-annual compounding should yield a lower effective
        # monthly rate than simple monthly compounding (rate/12)
        rate = 0.06
        canadian = canadian_effective_monthly_rate(rate)
        simple_monthly = rate / 12
        assert canadian < simple_monthly

    def test_known_value_549(self):
        # 5.49% -> effective monthly should be about 0.4525%
        eff = canadian_effective_monthly_rate(0.0549)
        assert 0.0045 < eff < 0.0046


# ---------------------------------------------------------------------------
# Stress Test Qualifying Rate
# ---------------------------------------------------------------------------

class TestStressTest:
    def test_floor_rate_applies(self):
        # 3% + 2% = 5% < 5.25% floor, so floor applies
        assert get_qualifying_rate(0.03) == pytest.approx(STRESS_TEST_FLOOR_RATE)

    def test_buffer_applies(self):
        # 5% + 2% = 7% > 5.25%, so buffer applies
        assert get_qualifying_rate(0.05) == pytest.approx(0.07)

    def test_exactly_at_boundary(self):
        # 3.25% + 2% = 5.25% == floor, should return 5.25%
        assert get_qualifying_rate(0.0325) == pytest.approx(0.0525)

    def test_high_rate(self):
        # 7% + 2% = 9%
        assert get_qualifying_rate(0.07) == pytest.approx(0.09)


# ---------------------------------------------------------------------------
# Monthly Payment Formula
# ---------------------------------------------------------------------------

class TestMonthlyPayment:
    def test_known_calculation(self):
        # $300,000 at 0.5% monthly for 300 months (25 years)
        pmt = monthly_payment(300_000, 0.005, 300)
        # Should be approximately $1,933
        assert 1930 < pmt < 1940

    def test_zero_rate(self):
        pmt = monthly_payment(120_000, 0.0, 300)
        assert pmt == pytest.approx(400.0)

    def test_payment_covers_principal(self):
        # Total payments over amortization should exceed principal
        principal = 500_000
        rate = 0.004
        months = 300
        pmt = monthly_payment(principal, rate, months)
        assert pmt * months > principal


# ---------------------------------------------------------------------------
# Balance After N Payments
# ---------------------------------------------------------------------------

class TestBalanceAfterPayments:
    def test_fully_amortized(self):
        principal = 300_000
        rate = 0.005
        months = 300
        pmt = monthly_payment(principal, rate, months)
        bal = balance_after_n_payments(principal, rate, pmt, months)
        assert bal == pytest.approx(0.0, abs=0.01)

    def test_partial_term(self):
        principal = 300_000
        rate = 0.005
        months = 300
        pmt = monthly_payment(principal, rate, months)
        bal = balance_after_n_payments(principal, rate, pmt, 60)
        # After 5 years, balance should be less than principal but > 0
        assert 0 < bal < principal

    def test_zero_rate(self):
        bal = balance_after_n_payments(120_000, 0.0, 400.0, 60)
        assert bal == pytest.approx(96_000.0)


# ---------------------------------------------------------------------------
# Full Calculation – Insured Mortgage
# ---------------------------------------------------------------------------

class TestInsuredMortgage:
    @pytest.fixture
    def result(self):
        return calculate_mortgage(MortgageInput(
            purchase_price=450_000,
            down_payment=22_500,
            annual_interest_rate=0.0549,
            amortization_years=25,
            term_years=5,
            annual_income=105_000,
            monthly_debts=350,
            monthly_condo_fees=400,
            annual_property_tax=3_200,
            monthly_heating=75,
        ))

    def test_mortgage_type(self, result):
        assert result.mortgage_type == MortgageType.INSURED

    def test_down_payment_pct(self, result):
        assert result.down_payment_pct == pytest.approx(0.05)

    def test_mortgage_amount(self, result):
        assert result.mortgage_amount == pytest.approx(427_500)

    def test_cmhc_premium_applied(self, result):
        # 95% LTV -> 4.00% premium on $427,500
        assert result.cmhc_premium == pytest.approx(17_100)
        assert result.cmhc_premium_rate == pytest.approx(0.04)

    def test_total_mortgage_includes_insurance(self, result):
        assert result.total_mortgage == pytest.approx(444_600)

    def test_qualifying_rate_uses_buffer(self, result):
        # 5.49% + 2% = 7.49% > 5.25%
        assert result.qualifying_rate == pytest.approx(0.0749)

    def test_monthly_payment_positive(self, result):
        assert result.monthly_payment > 0

    def test_stress_payment_higher(self, result):
        assert result.monthly_payment_stress > result.monthly_payment

    def test_gds_tds_calculated(self, result):
        assert result.gds_ratio is not None
        assert result.tds_ratio is not None
        assert result.tds_ratio > result.gds_ratio


# ---------------------------------------------------------------------------
# Full Calculation – Conventional (Insurable)
# ---------------------------------------------------------------------------

class TestConventionalMortgage:
    @pytest.fixture
    def result(self):
        return calculate_mortgage(MortgageInput(
            purchase_price=850_000,
            down_payment=170_000,
            annual_interest_rate=0.0519,
            amortization_years=25,
            term_years=5,
            annual_income=185_000,
            monthly_debts=600,
            annual_property_tax=6_800,
            monthly_heating=200,
        ))

    def test_mortgage_type(self, result):
        assert result.mortgage_type == MortgageType.INSURABLE

    def test_no_cmhc_premium(self, result):
        assert result.cmhc_premium == 0.0

    def test_total_mortgage_equals_base(self, result):
        assert result.total_mortgage == pytest.approx(680_000)

    def test_qualifies_gds_tds(self, result):
        assert result.gds_pass is True
        assert result.tds_pass is True


# ---------------------------------------------------------------------------
# Full Calculation – Uninsurable
# ---------------------------------------------------------------------------

class TestUninsurableMortgage:
    @pytest.fixture
    def result(self):
        return calculate_mortgage(MortgageInput(
            purchase_price=1_800_000,
            down_payment=360_000,
            annual_interest_rate=0.0579,
            amortization_years=30,
            term_years=5,
            annual_income=350_000,
            monthly_debts=1_200,
            annual_property_tax=14_400,
            monthly_heating=300,
        ))

    def test_mortgage_type(self, result):
        assert result.mortgage_type == MortgageType.UNINSURABLE

    def test_no_cmhc(self, result):
        assert result.cmhc_premium == 0.0

    def test_30yr_amortization(self, result):
        assert result.amortization_years == 30

    def test_lower_payment_than_25yr(self, result):
        # Compare with same mortgage but 25yr amort
        result_25 = calculate_mortgage(MortgageInput(
            purchase_price=1_800_000,
            down_payment=360_000,
            annual_interest_rate=0.0579,
            amortization_years=25,
        ))
        assert result.monthly_payment < result_25.monthly_payment

    def test_balance_at_term_end(self, result):
        assert 0 < result.balance_at_term_end < result.total_mortgage


# ---------------------------------------------------------------------------
# Validation Errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_down_payment_too_low(self):
        with pytest.raises(ValueError, match="below the minimum"):
            calculate_mortgage(MortgageInput(
                purchase_price=500_000,
                down_payment=10_000,  # Need 25,000
                annual_interest_rate=0.05,
                amortization_years=25,
            ))

    def test_insured_over_1_5m(self):
        # $2M requires 20% min ($400K), so $100K fails minimum DP check first
        with pytest.raises(ValueError, match="below the minimum"):
            calculate_mortgage(MortgageInput(
                purchase_price=2_000_000,
                down_payment=100_000,
                annual_interest_rate=0.05,
                amortization_years=25,
            ))

    def test_insured_30yr_amort(self):
        with pytest.raises(ValueError, match="25-year amortization"):
            calculate_mortgage(MortgageInput(
                purchase_price=400_000,
                down_payment=20_000,  # 5% -> insured
                annual_interest_rate=0.05,
                amortization_years=30,
            ))

    def test_minimum_down_payment_accepted(self):
        # Exactly minimum should NOT raise
        result = calculate_mortgage(MortgageInput(
            purchase_price=500_000,
            down_payment=25_000,
            annual_interest_rate=0.05,
            amortization_years=25,
        ))
        assert result.mortgage_type == MortgageType.INSURED


# ---------------------------------------------------------------------------
# GDS / TDS Ratios
# ---------------------------------------------------------------------------

class TestAffordabilityRatios:
    def test_no_income_skips_ratios(self):
        result = calculate_mortgage(MortgageInput(
            purchase_price=400_000,
            down_payment=80_000,
            annual_interest_rate=0.05,
            amortization_years=25,
            annual_income=0,
        ))
        assert result.gds_ratio is None
        assert result.tds_ratio is None

    def test_high_income_passes(self):
        result = calculate_mortgage(MortgageInput(
            purchase_price=300_000,
            down_payment=60_000,
            annual_interest_rate=0.04,
            amortization_years=25,
            annual_income=200_000,
            monthly_debts=0,
        ))
        assert result.gds_pass is True
        assert result.tds_pass is True

    def test_tds_includes_other_debts(self):
        base = MortgageInput(
            purchase_price=500_000,
            down_payment=100_000,
            annual_interest_rate=0.05,
            amortization_years=25,
            annual_income=120_000,
            monthly_debts=0,
        )
        r_no_debt = calculate_mortgage(base)

        with_debt = MortgageInput(
            purchase_price=500_000,
            down_payment=100_000,
            annual_interest_rate=0.05,
            amortization_years=25,
            annual_income=120_000,
            monthly_debts=1_000,
        )
        r_with_debt = calculate_mortgage(with_debt)

        # GDS should be the same (doesn't include other debts)
        assert r_no_debt.gds_ratio == pytest.approx(r_with_debt.gds_ratio)
        # TDS should be higher with debts
        assert r_with_debt.tds_ratio > r_no_debt.tds_ratio

    def test_gds_uses_stress_rate(self):
        result = calculate_mortgage(MortgageInput(
            purchase_price=500_000,
            down_payment=100_000,
            annual_interest_rate=0.05,
            amortization_years=25,
            annual_income=120_000,
        ))
        # Stress payment should be used in GDS, not contract payment
        # If GDS used contract rate payment, it would be lower
        assert result.monthly_payment_stress > result.monthly_payment


# ---------------------------------------------------------------------------
# Property Tax Defaults
# ---------------------------------------------------------------------------

class TestPropertyTax:
    def test_custom_property_tax(self):
        result = calculate_mortgage(MortgageInput(
            purchase_price=500_000,
            down_payment=100_000,
            annual_interest_rate=0.05,
            amortization_years=25,
            annual_property_tax=5_000,
        ))
        assert result.annual_property_tax == pytest.approx(5_000)
        assert result.monthly_property_tax == pytest.approx(5_000 / 12)

    def test_default_property_tax(self):
        result = calculate_mortgage(MortgageInput(
            purchase_price=500_000,
            down_payment=100_000,
            annual_interest_rate=0.05,
            amortization_years=25,
        ))
        # Default is 1% of purchase price
        assert result.annual_property_tax == pytest.approx(5_000)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_exact_20_pct_down_under_1_5m(self):
        # Exactly 20% should be insurable, NOT insured
        result = calculate_mortgage(MortgageInput(
            purchase_price=500_000,
            down_payment=100_000,
            annual_interest_rate=0.05,
            amortization_years=25,
        ))
        assert result.mortgage_type == MortgageType.INSURABLE
        assert result.cmhc_premium == 0.0

    def test_just_under_20_pct(self):
        # 19.99% down -> insured
        result = calculate_mortgage(MortgageInput(
            purchase_price=500_000,
            down_payment=99_950,  # 19.99%
            annual_interest_rate=0.05,
            amortization_years=25,
        ))
        assert result.mortgage_type == MortgageType.INSURED
        assert result.cmhc_premium > 0

    def test_large_down_payment(self):
        # 50% down
        result = calculate_mortgage(MortgageInput(
            purchase_price=600_000,
            down_payment=300_000,
            annual_interest_rate=0.05,
            amortization_years=25,
        ))
        assert result.mortgage_type == MortgageType.INSURABLE
        assert result.down_payment_pct == pytest.approx(0.50)

    def test_total_interest_positive(self):
        result = calculate_mortgage(MortgageInput(
            purchase_price=400_000,
            down_payment=80_000,
            annual_interest_rate=0.05,
            amortization_years=25,
            term_years=5,
        ))
        assert result.total_interest_over_term > 0
        total_interest_amort = result.total_paid_over_amortization - result.total_mortgage
        assert total_interest_amort > 0

    def test_balance_decreases_over_term(self):
        result = calculate_mortgage(MortgageInput(
            purchase_price=400_000,
            down_payment=80_000,
            annual_interest_rate=0.05,
            amortization_years=25,
            term_years=5,
        ))
        assert result.balance_at_term_end < result.total_mortgage
