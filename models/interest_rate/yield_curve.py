"""
Yield Curve Module for ALM Management System.

This module provides yield curve construction, interpolation, and scenario
analysis capabilities for institutional ALM. The yield curve is the
foundational input for all interest rate risk calculations.

Key Concepts:
-------------
- **Spot Rate (Zero Rate)**: The yield on a zero-coupon bond maturing at
  time t. Represents the pure time-value-of-money for that horizon.

- **Forward Rate**: The implied future interest rate between two future
  dates, derived from the spot curve via no-arbitrage.
  f(t1, t2) = [(1+s2)^t2 / (1+s1)^t1]^(1/(t2-t1)) - 1

- **Par Rate**: The coupon rate at which a bond prices at par given
  the spot curve. Used in bootstrapping.

- **Nelson-Siegel Model**: A parsimonious 4-parameter model for the
  yield curve that captures level, slope, and curvature dynamics.

- **Bootstrapping**: The process of extracting spot rates from observed
  par bond yields by iteratively solving for each zero rate.

Usage:
------
    from models.interest_rate.yield_curve import YieldCurve

    curve = YieldCurve.from_spot_rates(
        tenors=[1, 2, 5, 10, 30],
        rates=[0.025, 0.03, 0.035, 0.04, 0.045]
    )
    rate_7y = curve.spot_rate(7.0)
    fwd = curve.forward_rate(5.0, 10.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


class InterpolationMethod(Enum):
    """Interpolation method for yield curve construction."""
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"


@dataclass(frozen=True)
class NelsonSiegelParams:
    """
    Parameters for the Nelson-Siegel yield curve model.

    The Nelson-Siegel model decomposes the yield curve into three factors:

    r(t) = beta0 + beta1 * ((1-exp(-t/tau))/(t/tau))
                 + beta2 * ((1-exp(-t/tau))/(t/tau) - exp(-t/tau))

    Attributes:
        beta0: Level factor. Long-run equilibrium rate. As t -> inf, r(t) -> beta0.
               Represents the long-term expectations for interest rates.
        beta1: Slope factor. Short-end loading. beta0 + beta1 = instantaneous rate.
               Negative beta1 implies an upward-sloping curve (normal).
        beta2: Curvature factor. Controls the hump or trough shape.
               Positive beta2 creates a hump; negative creates a trough.
        tau:   Decay parameter. Controls where the curve's curvature is maximized.
               Larger tau shifts the hump to longer maturities.
    """
    beta0: float
    beta1: float
    beta2: float
    tau: float

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")


class YieldCurveError(Exception):
    """Base exception for yield curve errors."""
    pass


class YieldCurve:
    """
    Yield curve representation with interpolation and scenario analysis.

    Stores spot (zero-coupon) rates at discrete tenors and provides
    interpolation for arbitrary maturities. Supports construction from
    spot rates, par yields (via bootstrapping), or Nelson-Siegel parameters.

    Attributes:
        tenors: Array of maturities in years.
        rates: Array of spot rates (decimals) corresponding to each tenor.
        interpolation: Interpolation method used between observed tenors.

    Example:
        >>> curve = YieldCurve.from_spot_rates(
        ...     tenors=[1, 2, 5, 10, 30],
        ...     rates=[0.025, 0.03, 0.035, 0.04, 0.045]
        ... )
        >>> curve.spot_rate(7.0)
        0.0380  # interpolated
    """

    def __init__(
        self,
        tenors: NDArray[np.float64],
        rates: NDArray[np.float64],
        interpolation: InterpolationMethod = InterpolationMethod.LINEAR,
    ) -> None:
        """
        Initialize a YieldCurve.

        Use the class methods (from_spot_rates, from_par_yields, etc.)
        rather than calling this directly.

        Args:
            tenors: Sorted array of maturities in years.
            rates: Array of spot rates corresponding to each tenor.
            interpolation: Interpolation method for off-node rates.

        Raises:
            YieldCurveError: If inputs are invalid.
        """
        if len(tenors) < 2:
            raise YieldCurveError("At least 2 tenor points are required")
        if len(tenors) != len(rates):
            raise YieldCurveError(
                f"tenors length ({len(tenors)}) must match rates length ({len(rates)})"
            )
        if not np.all(np.diff(tenors) > 0):
            raise YieldCurveError("Tenors must be strictly increasing")
        if np.any(tenors <= 0):
            raise YieldCurveError("Tenors must be positive")

        self.tenors = tenors.copy()
        self.rates = rates.copy()
        self.interpolation = interpolation
        self._spline: Optional[CubicSpline] = None

        if interpolation == InterpolationMethod.CUBIC_SPLINE:
            self._spline = CubicSpline(self.tenors, self.rates)

    # ------------------------------------------------------------------
    # Construction class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_spot_rates(
        cls,
        tenors: list[float] | NDArray[np.float64],
        rates: list[float] | NDArray[np.float64],
        interpolation: InterpolationMethod = InterpolationMethod.LINEAR,
    ) -> YieldCurve:
        """
        Construct a yield curve from observed spot (zero-coupon) rates.

        Args:
            tenors: Maturities in years (e.g., [0.25, 0.5, 1, 2, 5, 10, 30]).
            rates: Spot rates as decimals (e.g., [0.02, 0.022, 0.025, ...]).
            interpolation: Interpolation method for off-node points.

        Returns:
            A YieldCurve instance.

        Example:
            >>> curve = YieldCurve.from_spot_rates(
            ...     tenors=[1, 2, 5, 10],
            ...     rates=[0.03, 0.035, 0.04, 0.045]
            ... )
        """
        return cls(
            tenors=np.asarray(tenors, dtype=np.float64),
            rates=np.asarray(rates, dtype=np.float64),
            interpolation=interpolation,
        )

    @classmethod
    def from_par_yields(
        cls,
        tenors: list[float] | NDArray[np.float64],
        par_yields: list[float] | NDArray[np.float64],
        frequency: int = 2,
        interpolation: InterpolationMethod = InterpolationMethod.LINEAR,
    ) -> YieldCurve:
        """
        Bootstrap a spot curve from par bond yields.

        Par yields are the coupon rates at which bonds of each maturity
        trade at par (price = 100). Bootstrapping extracts the implied
        zero-coupon (spot) rates iteratively.

        Bootstrapping Algorithm:
            For each maturity T with par yield c:
            1. Price all interim coupons using already-solved spot rates
            2. Solve for the spot rate at T such that the bond prices at par

            100 = sum(c/m * DF(t_i)) + (100 + c/m) * DF(T)

            where DF(t) = (1 + s(t))^(-t) is the discount factor.

        Args:
            tenors: Par bond maturities in years (must be multiples of 1/frequency).
            par_yields: Par coupon rates as decimals.
            frequency: Coupon payments per year (default: 2 for semi-annual).
            interpolation: Interpolation method for the resulting curve.

        Returns:
            A YieldCurve of bootstrapped spot rates.

        Raises:
            YieldCurveError: If bootstrapping fails to produce valid rates.

        Example:
            >>> # Semi-annual par yields
            >>> curve = YieldCurve.from_par_yields(
            ...     tenors=[0.5, 1, 2, 5, 10],
            ...     par_yields=[0.02, 0.025, 0.03, 0.035, 0.04]
            ... )
        """
        t_arr = np.asarray(tenors, dtype=np.float64)
        par_arr = np.asarray(par_yields, dtype=np.float64)

        if len(t_arr) < 2:
            raise YieldCurveError("At least 2 par yield points required for bootstrapping")
        if len(t_arr) != len(par_arr):
            raise YieldCurveError("tenors and par_yields must have the same length")

        spot_rates: dict[float, float] = {}
        period = 1.0 / frequency

        for i, (T, c) in enumerate(zip(t_arr, par_arr)):
            coupon = c / frequency  # Per-period coupon as fraction of par

            if i == 0:
                # First tenor: spot rate = par yield for a single-period instrument
                # 100 = (100 + coupon*100) / (1 + s)^T  =>  s = ((1 + coupon))^(1/T) - 1
                spot_rates[T] = ((1 + coupon)) ** (1.0 / T) - 1
            else:
                # Sum PV of interim coupons using known spot rates
                pv_coupons = 0.0
                t = period
                while t < T - 1e-10:
                    # Interpolate spot rate for this coupon date
                    s = cls._bootstrap_interpolate(t, spot_rates)
                    df = (1 + s) ** (-t)
                    pv_coupons += coupon * 100 * df
                    t += period

                # Solve for spot rate at T:
                # 100 = pv_coupons + (100 + coupon*100) * (1 + s_T)^(-T)
                final_payment = 100 + coupon * 100
                residual = 100 - pv_coupons

                if residual <= 0:
                    raise YieldCurveError(
                        f"Bootstrap failed at tenor {T}y: residual PV is non-positive"
                    )

                # (1 + s_T)^(-T) = residual / final_payment
                # s_T = (final_payment / residual)^(1/T) - 1
                s_T = (final_payment / residual) ** (1.0 / T) - 1
                spot_rates[T] = s_T

        sorted_tenors = sorted(spot_rates.keys())
        sorted_rates = [spot_rates[t] for t in sorted_tenors]

        return cls(
            tenors=np.array(sorted_tenors, dtype=np.float64),
            rates=np.array(sorted_rates, dtype=np.float64),
            interpolation=interpolation,
        )

    @classmethod
    def from_nelson_siegel(
        cls,
        params: NelsonSiegelParams,
        tenors: Optional[list[float] | NDArray[np.float64]] = None,
        interpolation: InterpolationMethod = InterpolationMethod.LINEAR,
    ) -> YieldCurve:
        """
        Generate a yield curve from Nelson-Siegel model parameters.

        Creates a discrete yield curve by evaluating the Nelson-Siegel
        formula at specified tenors.

        Args:
            params: Nelson-Siegel parameters (beta0, beta1, beta2, tau).
            tenors: Maturities to evaluate. Defaults to standard set
                    [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30].
            interpolation: Interpolation method for off-node points.

        Returns:
            A YieldCurve built from the Nelson-Siegel model.

        Example:
            >>> params = NelsonSiegelParams(beta0=0.045, beta1=-0.02, beta2=0.01, tau=1.5)
            >>> curve = YieldCurve.from_nelson_siegel(params)
        """
        if tenors is None:
            tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

        t_arr = np.asarray(tenors, dtype=np.float64)
        rates = cls._nelson_siegel_rates(t_arr, params)

        return cls(
            tenors=t_arr,
            rates=rates,
            interpolation=interpolation,
        )

    # ------------------------------------------------------------------
    # Rate queries
    # ------------------------------------------------------------------

    def spot_rate(self, tenor: float) -> float:
        """
        Get the interpolated spot rate at a given tenor.

        For tenors outside the observed range, flat extrapolation is used
        (the nearest endpoint rate is returned).

        Args:
            tenor: Maturity in years.

        Returns:
            Spot rate as decimal.

        Raises:
            YieldCurveError: If tenor is non-positive.
        """
        if tenor <= 0:
            raise YieldCurveError("Tenor must be positive")

        return float(self._interpolate(np.array([tenor]))[0])

    def spot_rates(self, tenors: list[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get interpolated spot rates at multiple tenors.

        Args:
            tenors: Array of maturities in years.

        Returns:
            Array of interpolated spot rates.
        """
        t_arr = np.asarray(tenors, dtype=np.float64)
        if np.any(t_arr <= 0):
            raise YieldCurveError("All tenors must be positive")
        return self._interpolate(t_arr)

    def discount_factor(self, tenor: float) -> float:
        """
        Get the discount factor for a given tenor.

        DF(t) = (1 + s(t))^(-t)

        Args:
            tenor: Maturity in years.

        Returns:
            Discount factor (value between 0 and 1 for positive rates).
        """
        rate = self.spot_rate(tenor)
        return float((1 + rate) ** (-tenor))

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Calculate the implied forward rate between two future dates.

        The forward rate f(t1, t2) is the rate that makes investing for
        t1 years and then reinvesting at f(t1,t2) until t2 equivalent to
        investing at the t2 spot rate for the full period (no-arbitrage).

        Formula:
            f(t1, t2) = [(1+s2)^t2 / (1+s1)^t1]^(1/(t2-t1)) - 1

        Args:
            t1: Start of forward period (years from now).
            t2: End of forward period (years from now).

        Returns:
            Forward rate as decimal.

        Raises:
            YieldCurveError: If t1 >= t2 or either tenor is non-positive.

        Example:
            >>> curve.forward_rate(2.0, 5.0)  # 3-year rate, 2 years forward
        """
        if t1 <= 0 or t2 <= 0:
            raise YieldCurveError("Both tenors must be positive")
        if t1 >= t2:
            raise YieldCurveError(f"t1 ({t1}) must be less than t2 ({t2})")

        s1 = self.spot_rate(t1)
        s2 = self.spot_rate(t2)

        # (1 + f)^(t2-t1) = (1+s2)^t2 / (1+s1)^t1
        growth_ratio = (1 + s2) ** t2 / (1 + s1) ** t1
        return float(growth_ratio ** (1.0 / (t2 - t1)) - 1)

    def forward_curve(
        self,
        start_tenors: list[float] | NDArray[np.float64],
        forward_period: float = 1.0,
    ) -> NDArray[np.float64]:
        """
        Calculate forward rates at multiple starting points.

        Computes the forward_period-year forward rate starting at each
        tenor in start_tenors.

        Args:
            start_tenors: Forward start dates in years.
            forward_period: Length of forward period in years (default: 1y).

        Returns:
            Array of forward rates.

        Example:
            >>> # 1-year forward rates starting at 1y, 2y, ..., 9y
            >>> fwd_rates = curve.forward_curve([1, 2, 3, 4, 5, 6, 7, 8, 9])
        """
        starts = np.asarray(start_tenors, dtype=np.float64)
        return np.array([
            self.forward_rate(t, t + forward_period)
            for t in starts
        ])

    # ------------------------------------------------------------------
    # Scenario analysis
    # ------------------------------------------------------------------

    def parallel_shift(self, basis_points: float) -> YieldCurve:
        """
        Create a new curve with a parallel shift applied.

        A parallel shift moves all rates up or down by the same amount.
        This is the simplest yield curve stress scenario and forms the
        basis for modified duration calculations.

        Args:
            basis_points: Shift in basis points (positive = rates up).
                          1 bp = 0.0001 in decimal.

        Returns:
            New YieldCurve with shifted rates.

        Example:
            >>> shocked_up = curve.parallel_shift(100)    # +100bp
            >>> shocked_down = curve.parallel_shift(-50)  # -50bp
        """
        shift = basis_points / 10_000
        return YieldCurve(
            tenors=self.tenors.copy(),
            rates=self.rates + shift,
            interpolation=self.interpolation,
        )

    def steepen(self, amount: float, pivot_tenor: float = 5.0) -> YieldCurve:
        """
        Apply a steepening twist to the curve.

        Steepening increases the spread between short and long rates.
        Rates below the pivot tenor decrease and rates above it increase,
        scaled linearly by distance from the pivot.

        In ALM context, steepening typically hurts institutions with
        long-duration assets funded by short-term liabilities.

        Args:
            amount: Total steepening in basis points (half applied
                    to each end relative to pivot).
            pivot_tenor: The tenor around which the twist pivots (default: 5y).

        Returns:
            New YieldCurve with steepened rates.

        Example:
            >>> # Steepen by 50bp total, pivoting at 5y
            >>> steep_curve = curve.steepen(50, pivot_tenor=5.0)
        """
        shift = amount / 10_000
        max_dist = max(
            pivot_tenor - self.tenors[0],
            self.tenors[-1] - pivot_tenor,
        )
        if max_dist == 0:
            return YieldCurve(
                tenors=self.tenors.copy(),
                rates=self.rates.copy(),
                interpolation=self.interpolation,
            )

        # Linear scaling: -shift/2 at short end, +shift/2 at long end
        adjustments = (self.tenors - pivot_tenor) / max_dist * (shift / 2)

        return YieldCurve(
            tenors=self.tenors.copy(),
            rates=self.rates + adjustments,
            interpolation=self.interpolation,
        )

    def flatten(self, amount: float, pivot_tenor: float = 5.0) -> YieldCurve:
        """
        Apply a flattening twist to the curve.

        Flattening reduces the spread between short and long rates.
        This is the inverse of steepening. In practice, the Fed hiking
        short rates often leads to curve flattening.

        Args:
            amount: Total flattening in basis points.
            pivot_tenor: The tenor around which the twist pivots (default: 5y).

        Returns:
            New YieldCurve with flattened rates.
        """
        return self.steepen(-amount, pivot_tenor)

    def invert(self, short_tenor: float = 2.0, long_tenor: float = 10.0) -> bool:
        """
        Check whether the curve is inverted between two tenors.

        An inverted yield curve (short rates > long rates) has historically
        been a leading indicator of recession. The most-watched spread is
        the 2s10s (2-year vs 10-year).

        Args:
            short_tenor: Short-end tenor (default: 2y).
            long_tenor: Long-end tenor (default: 10y).

        Returns:
            True if the curve is inverted (short rate > long rate).
        """
        return self.spot_rate(short_tenor) > self.spot_rate(long_tenor)

    def spread(self, short_tenor: float = 2.0, long_tenor: float = 10.0) -> float:
        """
        Calculate the spread between two points on the curve.

        A positive spread indicates a normal (upward-sloping) curve.
        A negative spread indicates inversion.

        Args:
            short_tenor: Short-end tenor (default: 2y).
            long_tenor: Long-end tenor (default: 10y).

        Returns:
            Spread in decimal (long rate - short rate).
        """
        return self.spot_rate(long_tenor) - self.spot_rate(short_tenor)

    def to_dict(self) -> dict[float, float]:
        """
        Export the curve as a {tenor: rate} dictionary.

        Useful for passing into key rate duration calculations.

        Returns:
            Dictionary mapping tenors to spot rates.
        """
        return {float(t): float(r) for t, r in zip(self.tenors, self.rates)}

    # ------------------------------------------------------------------
    # Nelson-Siegel fitting
    # ------------------------------------------------------------------

    def fit_nelson_siegel(self) -> NelsonSiegelParams:
        """
        Fit a Nelson-Siegel model to the current curve.

        Uses least-squares optimization to find the beta0, beta1, beta2,
        and tau parameters that best fit the observed spot rates.

        Returns:
            Fitted NelsonSiegelParams.

        Example:
            >>> params = curve.fit_nelson_siegel()
            >>> print(f"Level: {params.beta0:.4f}, Slope: {params.beta1:.4f}")
        """
        def objective(x: NDArray[np.float64]) -> float:
            beta0, beta1, beta2, tau = x
            if tau <= 0:
                return 1e10
            params = NelsonSiegelParams(beta0=beta0, beta1=beta1, beta2=beta2, tau=tau)
            fitted = self._nelson_siegel_rates(self.tenors, params)
            return float(np.sum((self.rates - fitted) ** 2))

        # Initial guesses from curve shape
        long_rate = float(self.rates[-1])
        short_rate = float(self.rates[0])

        x0 = np.array([long_rate, short_rate - long_rate, 0.0, 2.0])

        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 10_000, "xatol": 1e-10, "fatol": 1e-12},
        )

        return NelsonSiegelParams(
            beta0=result.x[0],
            beta1=result.x[1],
            beta2=result.x[2],
            tau=max(result.x[3], 1e-6),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interpolate(self, query_tenors: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Interpolate rates at arbitrary tenors.

        Uses the configured interpolation method. For tenors outside the
        observed range, flat extrapolation is applied.

        Args:
            query_tenors: Array of maturities to interpolate at.

        Returns:
            Array of interpolated rates.
        """
        if self.interpolation == InterpolationMethod.CUBIC_SPLINE:
            assert self._spline is not None
            # Clamp to observed range for extrapolation
            clamped = np.clip(query_tenors, self.tenors[0], self.tenors[-1])
            result = self._spline(clamped)
            # Flat extrapolation outside range
            result = np.where(query_tenors < self.tenors[0], self.rates[0], result)
            result = np.where(query_tenors > self.tenors[-1], self.rates[-1], result)
            return result

        # Linear interpolation with flat extrapolation
        return np.interp(query_tenors, self.tenors, self.rates)

    @staticmethod
    def _nelson_siegel_rates(
        tenors: NDArray[np.float64],
        params: NelsonSiegelParams,
    ) -> NDArray[np.float64]:
        """
        Evaluate the Nelson-Siegel formula at given tenors.

        r(t) = beta0 + beta1 * ((1 - exp(-t/tau)) / (t/tau))
                     + beta2 * ((1 - exp(-t/tau)) / (t/tau) - exp(-t/tau))

        Args:
            tenors: Array of maturities.
            params: Nelson-Siegel parameters.

        Returns:
            Array of rates.
        """
        t_over_tau = tenors / params.tau

        # Handle near-zero tenors to avoid division by zero
        # As t -> 0: (1 - exp(-x))/x -> 1
        with np.errstate(invalid="ignore", divide="ignore"):
            decay = np.where(
                t_over_tau < 1e-10,
                1.0,
                (1 - np.exp(-t_over_tau)) / t_over_tau,
            )
            exp_term = np.where(
                t_over_tau < 1e-10,
                1.0,
                np.exp(-t_over_tau),
            )

        return params.beta0 + params.beta1 * decay + params.beta2 * (decay - exp_term)

    @staticmethod
    def _bootstrap_interpolate(t: float, spot_rates: dict[float, float]) -> float:
        """
        Interpolate a spot rate during bootstrapping.

        Linear interpolation over already-solved spot rates. If t is before
        the first known tenor, the first rate is used (flat extrapolation).

        Args:
            t: Tenor to interpolate at.
            spot_rates: Dictionary of {tenor: spot_rate} already solved.

        Returns:
            Interpolated spot rate.
        """
        known_tenors = sorted(spot_rates.keys())
        known_rates = [spot_rates[k] for k in known_tenors]

        return float(np.interp(t, known_tenors, known_rates))
