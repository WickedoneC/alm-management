"""
Unit tests for Duration Calculator module.

These tests verify the correctness of duration calculations against
known analytical solutions and edge cases. Test cases include:
- Zero-coupon bonds (duration = maturity)
- Par bonds (known duration formulas)
- Edge cases and error handling
"""

import numpy as np
import pytest

from models.duration.duration_calculator import (
    CompoundingFrequency,
    DurationCalculator,
    DurationResult,
    InvalidCashFlowError,
    InvalidTimeError,
    InvalidYieldError,
    KeyRateDurationResult,
)


class TestMacaulayDuration:
    """Tests for Macaulay duration calculation."""

    def test_zero_coupon_bond_duration_equals_maturity(self):
        """For a zero-coupon bond, Macaulay duration equals time to maturity."""
        calc = DurationCalculator()

        # 5-year zero coupon bond, $1000 face
        cash_flows = [1000]
        times = [5.0]
        ytm = 0.05

        duration = calc.macaulay_duration(cash_flows, times, ytm)

        assert duration == pytest.approx(5.0, rel=1e-10)

    def test_zero_coupon_bond_various_maturities(self):
        """Zero-coupon bonds at various maturities."""
        calc = DurationCalculator()

        for maturity in [1, 2, 5, 10, 30]:
            duration = calc.macaulay_duration([1000], [maturity], 0.05)
            assert duration == pytest.approx(maturity, rel=1e-10)

    def test_coupon_bond_duration_less_than_maturity(self):
        """Coupon bond duration should be less than maturity."""
        calc = DurationCalculator()

        # 5-year, 5% annual coupon bond
        cash_flows = [50, 50, 50, 50, 1050]
        times = [1, 2, 3, 4, 5]
        ytm = 0.05

        duration = calc.macaulay_duration(cash_flows, times, ytm)

        assert duration < 5.0
        assert duration > 0

    def test_three_year_annual_coupon_bond(self):
        """
        Test a standard 3-year bond with 5% annual coupon at par.

        For a par bond with annual coupons:
        D = (1+y)/y - [(1+y) + T(c-y)] / [c((1+y)^T - 1) + y]

        where c = coupon rate, y = yield, T = maturity
        """
        calc = DurationCalculator()

        # 3-year, 5% coupon, priced at par (ytm = coupon rate)
        cash_flows = [50, 50, 1050]
        times = [1, 2, 3]
        ytm = 0.05

        duration = calc.macaulay_duration(cash_flows, times, ytm)

        # Manual calculation for verification:
        # PV1 = 50/1.05 = 47.619
        # PV2 = 50/1.05^2 = 45.351
        # PV3 = 1050/1.05^3 = 907.029
        # Total PV = 1000
        # Mac Duration = (1*47.619 + 2*45.351 + 3*907.029) / 1000 = 2.859
        expected = 2.859410430839002

        assert duration == pytest.approx(expected, rel=1e-6)

    def test_higher_coupon_reduces_duration(self):
        """Higher coupon rates should result in lower duration."""
        calc = DurationCalculator()
        times = [1, 2, 3, 4, 5]
        ytm = 0.05

        # Low coupon bond
        low_coupon_cf = [20, 20, 20, 20, 1020]
        low_coupon_dur = calc.macaulay_duration(low_coupon_cf, times, ytm)

        # High coupon bond
        high_coupon_cf = [100, 100, 100, 100, 1100]
        high_coupon_dur = calc.macaulay_duration(high_coupon_cf, times, ytm)

        assert low_coupon_dur > high_coupon_dur

    def test_higher_yield_reduces_duration(self):
        """Higher yields should result in slightly lower duration."""
        calc = DurationCalculator()
        cash_flows = [50, 50, 50, 50, 1050]
        times = [1, 2, 3, 4, 5]

        low_yield_dur = calc.macaulay_duration(cash_flows, times, 0.02)
        high_yield_dur = calc.macaulay_duration(cash_flows, times, 0.10)

        assert low_yield_dur > high_yield_dur

    def test_accepts_numpy_arrays(self):
        """Should accept numpy arrays as input."""
        calc = DurationCalculator()

        cash_flows = np.array([50, 50, 1050])
        times = np.array([1, 2, 3])

        duration = calc.macaulay_duration(cash_flows, times, 0.05)

        assert isinstance(duration, float)
        assert duration > 0


class TestModifiedDuration:
    """Tests for modified duration calculation."""

    def test_modified_duration_formula(self):
        """Modified duration = Macaulay duration / (1 + y/m)."""
        calc = DurationCalculator(frequency=CompoundingFrequency.ANNUAL)

        cash_flows = [50, 50, 1050]
        times = [1, 2, 3]
        ytm = 0.05

        mac_dur = calc.macaulay_duration(cash_flows, times, ytm)
        mod_dur = calc.modified_duration(cash_flows, times, ytm)

        expected_mod_dur = mac_dur / (1 + ytm)

        assert mod_dur == pytest.approx(expected_mod_dur, rel=1e-10)

    def test_semi_annual_compounding(self):
        """Test modified duration with semi-annual compounding."""
        calc = DurationCalculator(frequency=CompoundingFrequency.SEMI_ANNUAL)

        cash_flows = [25, 25, 25, 25, 25, 1025]
        times = [0.5, 1, 1.5, 2, 2.5, 3]
        ytm = 0.05

        mac_dur = calc.macaulay_duration(cash_flows, times, ytm)
        mod_dur = calc.modified_duration(cash_flows, times, ytm)

        expected_mod_dur = mac_dur / (1 + ytm / 2)

        assert mod_dur == pytest.approx(expected_mod_dur, rel=1e-10)

    def test_continuous_compounding(self):
        """For continuous compounding, modified = macaulay duration."""
        calc = DurationCalculator(frequency=CompoundingFrequency.CONTINUOUS)

        cash_flows = [50, 50, 1050]
        times = [1, 2, 3]
        ytm = 0.05

        mac_dur = calc.macaulay_duration(cash_flows, times, ytm)
        mod_dur = calc.modified_duration(cash_flows, times, ytm)

        assert mod_dur == pytest.approx(mac_dur, rel=1e-10)

    def test_modified_duration_always_less_than_macaulay(self):
        """Modified duration should be less than Macaulay for positive yields."""
        calc = DurationCalculator()

        cash_flows = [50, 50, 50, 50, 1050]
        times = [1, 2, 3, 4, 5]

        for ytm in [0.01, 0.05, 0.10, 0.15]:
            mac_dur = calc.macaulay_duration(cash_flows, times, ytm)
            mod_dur = calc.modified_duration(cash_flows, times, ytm)

            assert mod_dur < mac_dur


class TestConvexity:
    """Tests for convexity calculation."""

    def test_convexity_positive(self):
        """Convexity should be positive for standard bonds."""
        calc = DurationCalculator()

        cash_flows = [50, 50, 50, 50, 1050]
        times = [1, 2, 3, 4, 5]
        ytm = 0.05

        convexity = calc.convexity(cash_flows, times, ytm)

        assert convexity > 0

    def test_longer_maturity_higher_convexity(self):
        """Longer maturity bonds should have higher convexity."""
        calc = DurationCalculator()
        ytm = 0.05

        # 3-year bond
        cf_3y = [50, 50, 1050]
        times_3y = [1, 2, 3]
        conv_3y = calc.convexity(cf_3y, times_3y, ytm)

        # 10-year bond
        cf_10y = [50] * 9 + [1050]
        times_10y = list(range(1, 11))
        conv_10y = calc.convexity(cf_10y, times_10y, ytm)

        assert conv_10y > conv_3y

    def test_zero_coupon_convexity(self):
        """
        Zero-coupon bond convexity formula: C = T*(T+1)/(1+y)^2
        """
        calc = DurationCalculator()
        ytm = 0.05
        maturity = 5.0

        convexity = calc.convexity([1000], [maturity], ytm)

        expected = maturity * (maturity + 1) / (1 + ytm) ** 2

        assert convexity == pytest.approx(expected, rel=1e-10)


class TestDollarDuration:
    """Tests for dollar duration calculation."""

    def test_dollar_duration_formula(self):
        """Dollar duration = Modified duration × Price / 100."""
        calc = DurationCalculator()

        cash_flows = [50, 50, 1050]
        times = [1, 2, 3]
        ytm = 0.05

        dollar_dur = calc.dollar_duration(cash_flows, times, ytm)
        mod_dur = calc.modified_duration(cash_flows, times, ytm)

        # Price at par for 5% coupon, 5% yield
        price = 1000.0

        expected = mod_dur * price / 100

        assert dollar_dur == pytest.approx(expected, rel=1e-6)


class TestCalculateAllDurations:
    """Tests for the comprehensive duration calculation method."""

    def test_returns_duration_result(self):
        """Should return a DurationResult dataclass."""
        calc = DurationCalculator()

        result = calc.calculate_all_durations(
            cash_flows=[50, 50, 1050],
            times=[1, 2, 3],
            ytm=0.05
        )

        assert isinstance(result, DurationResult)
        assert hasattr(result, 'macaulay_duration')
        assert hasattr(result, 'modified_duration')
        assert hasattr(result, 'dollar_duration')
        assert hasattr(result, 'convexity')

    def test_all_values_positive(self):
        """All duration metrics should be positive for standard bonds."""
        calc = DurationCalculator()

        result = calc.calculate_all_durations(
            cash_flows=[50, 50, 50, 50, 1050],
            times=[1, 2, 3, 4, 5],
            ytm=0.05
        )

        assert result.macaulay_duration > 0
        assert result.modified_duration > 0
        assert result.dollar_duration > 0
        assert result.convexity > 0

    def test_consistency_with_individual_methods(self):
        """Results should match individual method calls."""
        calc = DurationCalculator()

        cash_flows = [50, 50, 1050]
        times = [1, 2, 3]
        ytm = 0.05

        result = calc.calculate_all_durations(cash_flows, times, ytm)

        assert result.macaulay_duration == pytest.approx(
            calc.macaulay_duration(cash_flows, times, ytm)
        )
        assert result.modified_duration == pytest.approx(
            calc.modified_duration(cash_flows, times, ytm)
        )
        assert result.convexity == pytest.approx(
            calc.convexity(cash_flows, times, ytm)
        )


class TestKeyRateDurations:
    """Tests for key rate duration calculation."""

    @pytest.fixture
    def flat_curve(self) -> dict[float, float]:
        """Flat yield curve at 5%."""
        return {
            0.25: 0.05, 0.5: 0.05, 1: 0.05, 2: 0.05,
            3: 0.05, 5: 0.05, 7: 0.05, 10: 0.05,
            20: 0.05, 30: 0.05
        }

    @pytest.fixture
    def upward_sloping_curve(self) -> dict[float, float]:
        """Upward sloping yield curve."""
        return {
            0.25: 0.02, 0.5: 0.022, 1: 0.025, 2: 0.03,
            3: 0.033, 5: 0.038, 7: 0.041, 10: 0.045,
            20: 0.048, 30: 0.05
        }

    def test_returns_key_rate_duration_result(self, flat_curve):
        """Should return a KeyRateDurationResult dataclass."""
        calc = DurationCalculator()

        result = calc.key_rate_durations(
            cash_flows=[50, 50, 1050],
            times=[1, 2, 3],
            spot_rates=flat_curve
        )

        assert isinstance(result, KeyRateDurationResult)
        assert hasattr(result, 'krd_2y')
        assert hasattr(result, 'krd_5y')
        assert hasattr(result, 'krd_10y')
        assert hasattr(result, 'krd_30y')
        assert hasattr(result, 'total_krd')

    def test_short_bond_concentrated_in_short_tenors(self, flat_curve):
        """3-year bond should have KRD concentrated in 2y and 5y buckets."""
        calc = DurationCalculator()

        # 3-year bond
        result = calc.key_rate_durations(
            cash_flows=[50, 50, 1050],
            times=[1, 2, 3],
            spot_rates=flat_curve
        )

        # Most sensitivity should be in 2y and 5y buckets
        short_end_krd = result.krd_2y + result.krd_5y
        long_end_krd = result.krd_10y + result.krd_30y

        assert short_end_krd > long_end_krd

    def test_long_bond_has_long_tenor_exposure(self, flat_curve):
        """30-year bond should have significant 30y KRD."""
        calc = DurationCalculator()

        # 30-year bond (simplified annual cash flows)
        cash_flows = [50] * 29 + [1050]
        times = list(range(1, 31))

        result = calc.key_rate_durations(
            cash_flows=cash_flows,
            times=times,
            spot_rates=flat_curve
        )

        # Should have exposure across the curve
        assert result.krd_30y > 0
        assert result.krd_10y > 0

    def test_total_krd_approximates_modified_duration(self, flat_curve):
        """Sum of KRDs should approximate modified duration for flat curve."""
        calc = DurationCalculator()

        cash_flows = [50, 50, 50, 50, 1050]
        times = [1, 2, 3, 4, 5]

        krd_result = calc.key_rate_durations(
            cash_flows=cash_flows,
            times=times,
            spot_rates=flat_curve
        )

        # For a flat curve, use the flat rate as YTM
        mod_dur = calc.modified_duration(cash_flows, times, ytm=0.05)

        # KRD total should be reasonably close to modified duration
        # (Not exact due to different calculation methodology)
        assert krd_result.total_krd == pytest.approx(mod_dur, rel=0.1)

    def test_missing_tenor_raises_error(self):
        """Should raise ValueError if spot_rates missing required tenor."""
        calc = DurationCalculator()

        # Missing 30y tenor
        incomplete_curve = {
            0.5: 0.02, 1: 0.025, 2: 0.03, 5: 0.035, 10: 0.04
        }

        with pytest.raises(ValueError, match="30"):
            calc.key_rate_durations(
                cash_flows=[50, 50, 1050],
                times=[1, 2, 3],
                spot_rates=incomplete_curve
            )


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_empty_cash_flows_raises_error(self):
        """Empty cash flows should raise InvalidCashFlowError."""
        calc = DurationCalculator()

        with pytest.raises(InvalidCashFlowError, match="empty"):
            calc.macaulay_duration([], [], 0.05)

    def test_mismatched_lengths_raises_error(self):
        """Mismatched cash_flows and times lengths should raise error."""
        calc = DurationCalculator()

        with pytest.raises(InvalidTimeError, match="length"):
            calc.macaulay_duration([50, 50, 1050], [1, 2], 0.05)

    def test_all_zero_cash_flows_raises_error(self):
        """All zero cash flows should raise InvalidCashFlowError."""
        calc = DurationCalculator()

        with pytest.raises(InvalidCashFlowError, match="non-zero"):
            calc.macaulay_duration([0, 0, 0], [1, 2, 3], 0.05)

    def test_negative_times_raises_error(self):
        """Negative times should raise InvalidTimeError."""
        calc = DurationCalculator()

        with pytest.raises(InvalidTimeError, match="negative"):
            calc.macaulay_duration([50, 50, 1050], [-1, 2, 3], 0.05)

    def test_non_increasing_times_raises_error(self):
        """Non-strictly-increasing times should raise error."""
        calc = DurationCalculator()

        with pytest.raises(InvalidTimeError, match="increasing"):
            calc.macaulay_duration([50, 50, 1050], [1, 3, 2], 0.05)

    def test_yield_less_than_minus_100_percent_raises_error(self):
        """Yield <= -100% should raise InvalidYieldError."""
        calc = DurationCalculator()

        with pytest.raises(InvalidYieldError):
            calc.macaulay_duration([50, 1050], [1, 2], -1.5)

    def test_negative_yield_accepted(self):
        """Negative yields (but > -100%) should be accepted."""
        calc = DurationCalculator()

        # Should not raise
        duration = calc.macaulay_duration([50, 1050], [1, 2], -0.01)

        assert duration > 0

    def test_zero_bump_size_raises_error(self):
        """Zero bump size should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DurationCalculator(bump_size=0)

    def test_negative_bump_size_raises_error(self):
        """Negative bump size should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DurationCalculator(bump_size=-0.0001)


class TestCompoundingFrequency:
    """Tests for compounding frequency handling."""

    def test_annual_frequency_value(self):
        """Annual frequency should have value 1."""
        assert CompoundingFrequency.ANNUAL.value == 1

    def test_semi_annual_frequency_value(self):
        """Semi-annual frequency should have value 2."""
        assert CompoundingFrequency.SEMI_ANNUAL.value == 2

    def test_quarterly_frequency_value(self):
        """Quarterly frequency should have value 4."""
        assert CompoundingFrequency.QUARTERLY.value == 4

    def test_monthly_frequency_value(self):
        """Monthly frequency should have value 12."""
        assert CompoundingFrequency.MONTHLY.value == 12

    def test_continuous_frequency_value(self):
        """Continuous frequency should have value 0."""
        assert CompoundingFrequency.CONTINUOUS.value == 0


class TestDataclassImmutability:
    """Tests for dataclass immutability."""

    def test_duration_result_is_frozen(self):
        """DurationResult should be immutable."""
        result = DurationResult(
            macaulay_duration=3.0,
            modified_duration=2.85,
            dollar_duration=28.5,
            convexity=12.0
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.macaulay_duration = 5.0

    def test_key_rate_duration_result_is_frozen(self):
        """KeyRateDurationResult should be immutable."""
        result = KeyRateDurationResult(
            krd_2y=0.5,
            krd_5y=1.0,
            krd_10y=0.8,
            krd_30y=0.2,
            total_krd=2.5
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.krd_2y = 1.0
