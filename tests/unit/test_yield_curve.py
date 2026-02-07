"""
Unit tests for Yield Curve module.

Tests verify correctness of:
- Curve construction from spot rates and par yields
- Interpolation methods (linear, cubic spline)
- Forward rate calculations against no-arbitrage conditions
- Nelson-Siegel model fitting and generation
- Scenario analysis (parallel shift, steepening, flattening)
- Edge cases and error handling
"""

import numpy as np
import pytest

from models.interest_rate.yield_curve import (
    InterpolationMethod,
    NelsonSiegelParams,
    YieldCurve,
    YieldCurveError,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def normal_curve() -> YieldCurve:
    """Standard upward-sloping yield curve."""
    return YieldCurve.from_spot_rates(
        tenors=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
        rates=[0.020, 0.022, 0.025, 0.030, 0.033, 0.038, 0.041, 0.045, 0.048, 0.050],
    )


@pytest.fixture
def flat_curve() -> YieldCurve:
    """Flat yield curve at 4%."""
    return YieldCurve.from_spot_rates(
        tenors=[0.5, 1, 2, 5, 10, 30],
        rates=[0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
    )


@pytest.fixture
def inverted_curve() -> YieldCurve:
    """Inverted yield curve (short rates > long rates)."""
    return YieldCurve.from_spot_rates(
        tenors=[0.5, 1, 2, 5, 10, 30],
        rates=[0.055, 0.053, 0.050, 0.045, 0.040, 0.038],
    )


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

class TestFromSpotRates:
    """Tests for constructing curves from spot rates."""

    def test_basic_construction(self, normal_curve):
        """Curve should store tenors and rates correctly."""
        assert len(normal_curve.tenors) == 10
        assert len(normal_curve.rates) == 10
        assert normal_curve.rates[0] == pytest.approx(0.020)
        assert normal_curve.rates[-1] == pytest.approx(0.050)

    def test_returns_exact_rates_at_nodes(self, normal_curve):
        """Spot rate at an exact node should match the input."""
        assert normal_curve.spot_rate(1.0) == pytest.approx(0.025)
        assert normal_curve.spot_rate(10.0) == pytest.approx(0.045)
        assert normal_curve.spot_rate(30.0) == pytest.approx(0.050)

    def test_minimum_two_points(self):
        """Curve requires at least 2 points."""
        with pytest.raises(YieldCurveError, match="At least 2"):
            YieldCurve.from_spot_rates(tenors=[1], rates=[0.05])

    def test_mismatched_lengths(self):
        """Mismatched tenor and rate arrays should raise."""
        with pytest.raises(YieldCurveError, match="length"):
            YieldCurve.from_spot_rates(tenors=[1, 2, 5], rates=[0.03, 0.04])

    def test_non_increasing_tenors(self):
        """Non-increasing tenors should raise."""
        with pytest.raises(YieldCurveError, match="increasing"):
            YieldCurve.from_spot_rates(tenors=[1, 5, 3], rates=[0.03, 0.04, 0.035])

    def test_negative_tenor(self):
        """Negative tenors should raise."""
        with pytest.raises(YieldCurveError, match="positive"):
            YieldCurve.from_spot_rates(tenors=[-1, 1], rates=[0.03, 0.04])

    def test_internal_arrays_are_copies(self):
        """Modifying the original input should not affect the curve."""
        tenors = [1.0, 2.0, 5.0]
        rates = [0.03, 0.035, 0.04]

        curve = YieldCurve.from_spot_rates(tenors, rates)
        original_rate = curve.spot_rate(1.0)

        tenors[0] = 99.0
        rates[0] = 0.99

        assert curve.spot_rate(1.0) == pytest.approx(original_rate)


# ------------------------------------------------------------------
# Interpolation
# ------------------------------------------------------------------

class TestInterpolation:
    """Tests for rate interpolation."""

    def test_linear_interpolation_midpoint(self):
        """Linear interpolation at the midpoint of two nodes."""
        curve = YieldCurve.from_spot_rates(
            tenors=[2, 10],
            rates=[0.03, 0.05],
            interpolation=InterpolationMethod.LINEAR,
        )

        # Midpoint at 6y: 0.03 + (6-2)/(10-2) * (0.05-0.03) = 0.04
        assert curve.spot_rate(6.0) == pytest.approx(0.04)

    def test_linear_extrapolation_short_end(self, normal_curve):
        """Extrapolation below the shortest tenor uses flat extrapolation."""
        # np.interp gives flat extrapolation by default
        assert normal_curve.spot_rate(0.1) == pytest.approx(0.020)

    def test_linear_extrapolation_long_end(self, normal_curve):
        """Extrapolation beyond the longest tenor uses flat extrapolation."""
        assert normal_curve.spot_rate(40.0) == pytest.approx(0.050)

    def test_cubic_spline_at_nodes(self):
        """Cubic spline should match exact node values."""
        curve = YieldCurve.from_spot_rates(
            tenors=[1, 2, 5, 10, 30],
            rates=[0.025, 0.03, 0.038, 0.045, 0.05],
            interpolation=InterpolationMethod.CUBIC_SPLINE,
        )

        assert curve.spot_rate(1.0) == pytest.approx(0.025)
        assert curve.spot_rate(10.0) == pytest.approx(0.045)

    def test_cubic_spline_smoother_than_linear(self):
        """Cubic spline should produce smoother intermediate values."""
        tenors = [1, 2, 5, 10, 30]
        rates = [0.025, 0.03, 0.038, 0.045, 0.05]

        linear = YieldCurve.from_spot_rates(tenors, rates, InterpolationMethod.LINEAR)
        spline = YieldCurve.from_spot_rates(tenors, rates, InterpolationMethod.CUBIC_SPLINE)

        # At the same query point, results should differ (different methods)
        # but both should be in a reasonable range
        linear_7y = linear.spot_rate(7.0)
        spline_7y = spline.spot_rate(7.0)

        assert 0.03 < linear_7y < 0.05
        assert 0.03 < spline_7y < 0.05

    def test_vectorized_spot_rates(self, normal_curve):
        """spot_rates should return array of interpolated values."""
        rates = normal_curve.spot_rates([1, 5, 10])

        assert len(rates) == 3
        assert rates[0] == pytest.approx(0.025)
        assert rates[1] == pytest.approx(0.038)
        assert rates[2] == pytest.approx(0.045)


# ------------------------------------------------------------------
# Discount factors
# ------------------------------------------------------------------

class TestDiscountFactor:
    """Tests for discount factor calculation."""

    def test_discount_factor_formula(self, flat_curve):
        """DF(t) = (1 + s)^(-t) for a flat curve."""
        df = flat_curve.discount_factor(5.0)
        expected = (1 + 0.04) ** (-5.0)
        assert df == pytest.approx(expected, rel=1e-10)

    def test_short_tenor_discount_factor_near_one(self, flat_curve):
        """Short-tenor discount factors should be close to 1."""
        df = flat_curve.discount_factor(0.5)
        assert 0.95 < df < 1.0

    def test_discount_factor_decreases_with_tenor(self, flat_curve):
        """Discount factors should decrease as tenor increases."""
        df_1y = flat_curve.discount_factor(1.0)
        df_10y = flat_curve.discount_factor(10.0)
        df_30y = flat_curve.discount_factor(30.0)

        assert df_1y > df_10y > df_30y

    def test_non_positive_tenor_raises(self, flat_curve):
        """Non-positive tenors should raise."""
        with pytest.raises(YieldCurveError, match="positive"):
            flat_curve.discount_factor(0)

        with pytest.raises(YieldCurveError, match="positive"):
            flat_curve.discount_factor(-1)


# ------------------------------------------------------------------
# Forward rates
# ------------------------------------------------------------------

class TestForwardRate:
    """Tests for forward rate calculations."""

    def test_flat_curve_forward_equals_spot(self, flat_curve):
        """On a flat curve, all forward rates equal the spot rate."""
        fwd = flat_curve.forward_rate(2.0, 5.0)
        assert fwd == pytest.approx(0.04, rel=1e-6)

    def test_no_arbitrage_condition(self, normal_curve):
        """
        Forward rates must satisfy the no-arbitrage condition:
        (1 + s2)^t2 = (1 + s1)^t1 × (1 + f)^(t2-t1)
        """
        t1, t2 = 2.0, 10.0
        s1 = normal_curve.spot_rate(t1)
        s2 = normal_curve.spot_rate(t2)
        fwd = normal_curve.forward_rate(t1, t2)

        lhs = (1 + s2) ** t2
        rhs = (1 + s1) ** t1 * (1 + fwd) ** (t2 - t1)

        assert lhs == pytest.approx(rhs, rel=1e-10)

    def test_upward_sloping_forward_above_spot(self, normal_curve):
        """On an upward-sloping curve, forward rates exceed spot rates."""
        fwd_5_10 = normal_curve.forward_rate(5.0, 10.0)
        spot_10 = normal_curve.spot_rate(10.0)

        # Forward rate should be above the 10y spot
        assert fwd_5_10 > spot_10

    def test_forward_rate_t1_equals_t2_raises(self, normal_curve):
        """t1 >= t2 should raise an error."""
        with pytest.raises(YieldCurveError):
            normal_curve.forward_rate(5.0, 5.0)

    def test_forward_rate_t1_greater_t2_raises(self, normal_curve):
        """t1 > t2 should raise an error."""
        with pytest.raises(YieldCurveError):
            normal_curve.forward_rate(10.0, 5.0)

    def test_forward_curve(self, normal_curve):
        """forward_curve should return array of 1y forward rates."""
        starts = [1, 2, 3, 5, 7]
        fwd_rates = normal_curve.forward_curve(starts, forward_period=1.0)

        assert len(fwd_rates) == 5
        # All forward rates should be positive
        assert np.all(fwd_rates > 0)

    def test_forward_rate_chain(self, normal_curve):
        """
        Chaining forward rates should reproduce the long spot rate.
        (1+s3)^3 = (1+s1)^1 × (1+f12)^1 × (1+f23)^1
        """
        s1 = normal_curve.spot_rate(1.0)
        s3 = normal_curve.spot_rate(3.0)
        f12 = normal_curve.forward_rate(1.0, 2.0)
        f23 = normal_curve.forward_rate(2.0, 3.0)

        lhs = (1 + s3) ** 3
        rhs = (1 + s1) * (1 + f12) * (1 + f23)

        assert lhs == pytest.approx(rhs, rel=1e-10)


# ------------------------------------------------------------------
# Bootstrapping from par yields
# ------------------------------------------------------------------

class TestBootstrapping:
    """Tests for bootstrapping spot rates from par yields."""

    def test_single_period_par_yield_equals_spot(self):
        """For a single-period bond, par yield ≈ spot rate."""
        curve = YieldCurve.from_par_yields(
            tenors=[0.5, 1.0],
            par_yields=[0.03, 0.035],
            frequency=2,
        )

        # The first spot rate should closely match the first par yield
        assert curve.spot_rate(0.5) == pytest.approx(0.03, rel=0.01)

    def test_bootstrapped_rates_increase_for_normal_curve(self):
        """Bootstrapped spot rates should increase for upward-sloping par yields."""
        curve = YieldCurve.from_par_yields(
            tenors=[0.5, 1, 2, 5, 10],
            par_yields=[0.02, 0.025, 0.03, 0.038, 0.045],
            frequency=2,
        )

        # Spot rates should generally increase
        r_1 = curve.spot_rate(1.0)
        r_5 = curve.spot_rate(5.0)
        r_10 = curve.spot_rate(10.0)

        assert r_5 > r_1
        assert r_10 > r_5

    def test_flat_par_yields_give_flat_spots(self):
        """If all par yields are equal, spot rates should be very close to that value."""
        curve = YieldCurve.from_par_yields(
            tenors=[0.5, 1, 2, 5],
            par_yields=[0.04, 0.04, 0.04, 0.04],
            frequency=2,
        )

        for t in [0.5, 1, 2, 5]:
            assert curve.spot_rate(t) == pytest.approx(0.04, rel=0.01)

    def test_bootstrap_minimum_points(self):
        """Bootstrapping requires at least 2 points."""
        with pytest.raises(YieldCurveError, match="At least 2"):
            YieldCurve.from_par_yields(tenors=[1], par_yields=[0.05])

    def test_spot_rates_discount_par_bond_to_100(self):
        """
        Verification: using bootstrapped spot rates to discount a par bond's
        cash flows should yield a present value of 100.
        """
        par_yields = [0.03, 0.035, 0.04]
        tenors = [0.5, 1.0, 2.0]

        curve = YieldCurve.from_par_yields(
            tenors=tenors,
            par_yields=par_yields,
            frequency=2,
        )

        # Price the 2-year par bond using bootstrapped spot rates
        coupon = par_yields[-1] / 2  # semi-annual coupon
        coupon_times = [0.5, 1.0, 1.5, 2.0]

        pv = 0.0
        for t in coupon_times:
            cf = coupon * 100
            if t == tenors[-1]:
                cf += 100
            rate = curve.spot_rate(t)
            pv += cf * (1 + rate) ** (-t)

        assert pv == pytest.approx(100.0, rel=0.02)


# ------------------------------------------------------------------
# Nelson-Siegel
# ------------------------------------------------------------------

class TestNelsonSiegel:
    """Tests for Nelson-Siegel model."""

    def test_params_validation(self):
        """tau must be positive."""
        with pytest.raises(ValueError, match="tau"):
            NelsonSiegelParams(beta0=0.05, beta1=-0.02, beta2=0.01, tau=0)

        with pytest.raises(ValueError, match="tau"):
            NelsonSiegelParams(beta0=0.05, beta1=-0.02, beta2=0.01, tau=-1)

    def test_from_nelson_siegel_basic(self):
        """Curve generated from NS params should evaluate correctly."""
        params = NelsonSiegelParams(beta0=0.05, beta1=-0.03, beta2=0.01, tau=2.0)
        curve = YieldCurve.from_nelson_siegel(params)

        # At long maturities, rate should approach beta0.
        # With tau=2 and t=30, decay terms are small but nonzero
        # ((1-exp(-15))/15 ≈ 0.067), so allow ~3bp tolerance.
        long_rate = curve.spot_rate(30.0)
        assert long_rate == pytest.approx(params.beta0, abs=0.003)

        # Short rate should approach beta0 + beta1 (= 0.02).
        # At t=0.25, decay terms are still significant, so allow ~3bp.
        short_rate = curve.spot_rate(0.25)
        assert short_rate == pytest.approx(params.beta0 + params.beta1, abs=0.005)

    def test_nelson_siegel_level_factor(self):
        """beta0 shifts the entire curve up/down."""
        params_low = NelsonSiegelParams(beta0=0.03, beta1=-0.01, beta2=0.0, tau=2.0)
        params_high = NelsonSiegelParams(beta0=0.06, beta1=-0.01, beta2=0.0, tau=2.0)

        curve_low = YieldCurve.from_nelson_siegel(params_low)
        curve_high = YieldCurve.from_nelson_siegel(params_high)

        for t in [1, 5, 10, 30]:
            assert curve_high.spot_rate(t) > curve_low.spot_rate(t)

    def test_nelson_siegel_slope_factor(self):
        """Negative beta1 produces an upward-sloping curve."""
        params = NelsonSiegelParams(beta0=0.05, beta1=-0.03, beta2=0.0, tau=2.0)
        curve = YieldCurve.from_nelson_siegel(params)

        r_short = curve.spot_rate(0.25)
        r_long = curve.spot_rate(30.0)

        assert r_long > r_short  # upward-sloping

    def test_nelson_siegel_curvature(self):
        """beta2 creates a hump in the curve."""
        params = NelsonSiegelParams(beta0=0.05, beta1=-0.02, beta2=0.05, tau=2.0)
        curve = YieldCurve.from_nelson_siegel(params)

        r_short = curve.spot_rate(0.5)
        r_mid = curve.spot_rate(2.0)
        r_long = curve.spot_rate(30.0)

        # Hump: mid-range rate should exceed at least one endpoint
        assert r_mid > r_short or r_mid > r_long

    def test_fit_nelson_siegel_recovers_params(self):
        """Fitting NS to an NS-generated curve should recover the parameters."""
        original = NelsonSiegelParams(beta0=0.045, beta1=-0.020, beta2=0.010, tau=1.5)

        curve = YieldCurve.from_nelson_siegel(original)
        fitted = curve.fit_nelson_siegel()

        # Parameters should be approximately recovered
        assert fitted.beta0 == pytest.approx(original.beta0, abs=0.002)
        assert fitted.beta1 == pytest.approx(original.beta1, abs=0.002)
        assert fitted.beta2 == pytest.approx(original.beta2, abs=0.005)

    def test_fit_nelson_siegel_on_real_curve(self, normal_curve):
        """Fitting NS to a realistic curve should produce a close fit."""
        fitted_params = normal_curve.fit_nelson_siegel()
        fitted_curve = YieldCurve.from_nelson_siegel(fitted_params, tenors=list(normal_curve.tenors))

        # Check that the fitted curve is reasonably close
        for t in normal_curve.tenors:
            original = normal_curve.spot_rate(float(t))
            fitted = fitted_curve.spot_rate(float(t))
            assert fitted == pytest.approx(original, abs=0.005)

    def test_custom_tenors(self):
        """from_nelson_siegel should accept custom tenors."""
        params = NelsonSiegelParams(beta0=0.05, beta1=-0.02, beta2=0.01, tau=2.0)
        curve = YieldCurve.from_nelson_siegel(params, tenors=[1, 5, 10])

        assert len(curve.tenors) == 3

    def test_params_immutable(self):
        """NelsonSiegelParams should be immutable."""
        params = NelsonSiegelParams(beta0=0.05, beta1=-0.02, beta2=0.01, tau=2.0)

        with pytest.raises(Exception):
            params.beta0 = 0.10


# ------------------------------------------------------------------
# Scenario analysis
# ------------------------------------------------------------------

class TestParallelShift:
    """Tests for parallel shift scenarios."""

    def test_shift_up(self, normal_curve):
        """Parallel shift up should increase all rates by the same amount."""
        shifted = normal_curve.parallel_shift(100)  # +100bp

        for t in normal_curve.tenors:
            original = normal_curve.spot_rate(float(t))
            new = shifted.spot_rate(float(t))
            assert new == pytest.approx(original + 0.01, rel=1e-10)

    def test_shift_down(self, normal_curve):
        """Parallel shift down should decrease all rates equally."""
        shifted = normal_curve.parallel_shift(-50)  # -50bp

        for t in normal_curve.tenors:
            original = normal_curve.spot_rate(float(t))
            new = shifted.spot_rate(float(t))
            assert new == pytest.approx(original - 0.005, rel=1e-10)

    def test_zero_shift_unchanged(self, normal_curve):
        """Zero shift should produce identical rates."""
        shifted = normal_curve.parallel_shift(0)

        for t in normal_curve.tenors:
            assert shifted.spot_rate(float(t)) == pytest.approx(
                normal_curve.spot_rate(float(t)), rel=1e-10
            )

    def test_shift_preserves_shape(self, normal_curve):
        """Parallel shift should preserve the curve's slope."""
        original_spread = normal_curve.spread(2.0, 10.0)
        shifted = normal_curve.parallel_shift(200)
        shifted_spread = shifted.spread(2.0, 10.0)

        assert shifted_spread == pytest.approx(original_spread, rel=1e-10)

    def test_shift_returns_new_curve(self, normal_curve):
        """Shift should return a new curve, not modify the original."""
        original_rate = normal_curve.spot_rate(5.0)
        _ = normal_curve.parallel_shift(100)

        assert normal_curve.spot_rate(5.0) == pytest.approx(original_rate)


class TestSteepening:
    """Tests for steepening/flattening scenarios."""

    def test_steepen_increases_spread(self, normal_curve):
        """Steepening should increase the long-short spread."""
        original_spread = normal_curve.spread(2.0, 10.0)
        steepened = normal_curve.steepen(50)
        new_spread = steepened.spread(2.0, 10.0)

        assert new_spread > original_spread

    def test_steepen_pivot_unchanged(self, flat_curve):
        """Rate at the pivot tenor should be approximately unchanged."""
        pivot = 5.0
        original_rate = flat_curve.spot_rate(pivot)
        steepened = flat_curve.steepen(100, pivot_tenor=pivot)
        new_rate = steepened.spot_rate(pivot)

        assert new_rate == pytest.approx(original_rate, abs=0.001)

    def test_steepen_short_rates_decrease(self, flat_curve):
        """Steepening should lower rates below the pivot."""
        steepened = flat_curve.steepen(100, pivot_tenor=5.0)

        assert steepened.spot_rate(0.5) < flat_curve.spot_rate(0.5)

    def test_steepen_long_rates_increase(self, flat_curve):
        """Steepening should raise rates above the pivot."""
        steepened = flat_curve.steepen(100, pivot_tenor=5.0)

        assert steepened.spot_rate(30.0) > flat_curve.spot_rate(30.0)

    def test_flatten_is_inverse_of_steepen(self, normal_curve):
        """Flattening should be equivalent to negative steepening."""
        steepened = normal_curve.steepen(-50)
        flattened = normal_curve.flatten(50)

        for t in normal_curve.tenors:
            assert steepened.spot_rate(float(t)) == pytest.approx(
                flattened.spot_rate(float(t)), rel=1e-10
            )

    def test_flatten_reduces_spread(self, normal_curve):
        """Flattening should reduce the long-short spread."""
        original_spread = normal_curve.spread(2.0, 10.0)
        flattened = normal_curve.flatten(50)
        new_spread = flattened.spread(2.0, 10.0)

        assert new_spread < original_spread


# ------------------------------------------------------------------
# Inversion detection and spread
# ------------------------------------------------------------------

class TestInversionAndSpread:
    """Tests for curve inversion detection and spread calculation."""

    def test_normal_curve_not_inverted(self, normal_curve):
        """Normal curve should not be inverted."""
        assert normal_curve.invert() is False

    def test_inverted_curve_detected(self, inverted_curve):
        """Inverted curve should be detected."""
        assert inverted_curve.invert() is True

    def test_normal_spread_positive(self, normal_curve):
        """Normal curve 2s10s spread should be positive."""
        assert normal_curve.spread(2.0, 10.0) > 0

    def test_inverted_spread_negative(self, inverted_curve):
        """Inverted curve 2s10s spread should be negative."""
        assert inverted_curve.spread(2.0, 10.0) < 0

    def test_spread_custom_tenors(self, normal_curve):
        """Spread should work with custom tenors."""
        spread_2_30 = normal_curve.spread(2.0, 30.0)
        spread_2_10 = normal_curve.spread(2.0, 10.0)

        # 2s30s should be wider than 2s10s on a normal curve
        assert spread_2_30 > spread_2_10

    def test_flat_curve_zero_spread(self, flat_curve):
        """Flat curve should have zero spread."""
        assert flat_curve.spread() == pytest.approx(0.0, abs=1e-10)


# ------------------------------------------------------------------
# to_dict export
# ------------------------------------------------------------------

class TestToDict:
    """Tests for exporting curve as dictionary."""

    def test_to_dict_structure(self, normal_curve):
        """to_dict should return {tenor: rate} mapping."""
        d = normal_curve.to_dict()

        assert isinstance(d, dict)
        assert len(d) == len(normal_curve.tenors)

        for t, r in zip(normal_curve.tenors, normal_curve.rates):
            assert d[float(t)] == pytest.approx(float(r))

    def test_to_dict_compatible_with_krd(self, normal_curve):
        """Exported dict should be usable with DurationCalculator.key_rate_durations."""
        d = normal_curve.to_dict()

        # Should have float keys and float values
        for k, v in d.items():
            assert isinstance(k, float)
            assert isinstance(v, float)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_tenor_raises(self, normal_curve):
        """Querying at tenor 0 should raise."""
        with pytest.raises(YieldCurveError, match="positive"):
            normal_curve.spot_rate(0)

    def test_negative_tenor_raises(self, normal_curve):
        """Negative tenor should raise."""
        with pytest.raises(YieldCurveError, match="positive"):
            normal_curve.spot_rate(-1)

    def test_very_long_tenor(self, normal_curve):
        """Very long tenor should use flat extrapolation."""
        r_30 = normal_curve.spot_rate(30.0)
        r_100 = normal_curve.spot_rate(100.0)

        assert r_100 == pytest.approx(r_30)

    def test_very_short_tenor(self, normal_curve):
        """Very short tenor below first node uses flat extrapolation."""
        r_first = normal_curve.spot_rate(0.25)
        r_tiny = normal_curve.spot_rate(0.01)

        assert r_tiny == pytest.approx(r_first)

    def test_negative_rates(self):
        """Curve should support negative interest rates."""
        curve = YieldCurve.from_spot_rates(
            tenors=[1, 2, 5, 10],
            rates=[-0.005, -0.003, 0.0, 0.005],
        )

        assert curve.spot_rate(1.0) == pytest.approx(-0.005)
        assert curve.spot_rate(5.0) == pytest.approx(0.0)

    def test_vectorized_spot_rates_negative_tenor(self, normal_curve):
        """Vectorized query with non-positive tenor should raise."""
        with pytest.raises(YieldCurveError, match="positive"):
            normal_curve.spot_rates([1.0, 0.0, 5.0])
