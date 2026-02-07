"""
Duration Calculator Module for ALM Management System.

This module provides institutional-grade duration calculations for fixed income
securities, essential for Asset-Liability Management (ALM). Duration measures
the sensitivity of a bond's price to changes in interest rates.

Key Concepts:
-------------
- **Macaulay Duration**: The weighted average time until cash flows are received,
  where weights are the present values of cash flows. Expressed in years.

- **Modified Duration**: Measures the percentage change in bond price for a 1%
  change in yield. It's Macaulay Duration adjusted for the compounding frequency.
  Modified Duration = Macaulay Duration / (1 + yield/frequency)

- **Key Rate Duration (KRD)**: Measures sensitivity to changes at specific points
  on the yield curve (e.g., 2y, 5y, 10y, 30y). Essential for understanding
  exposure to non-parallel yield curve shifts.

Usage:
------
    from models.duration.duration_calculator import DurationCalculator

    calc = DurationCalculator()

    # Calculate durations for a bond
    mac_dur = calc.macaulay_duration(
        cash_flows=[50, 50, 50, 1050],
        times=[1, 2, 3, 4],
        ytm=0.05
    )
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class CompoundingFrequency(Enum):
    """Compounding frequency for yield calculations."""
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    CONTINUOUS = 0  # Special case for continuous compounding


@dataclass(frozen=True)
class DurationResult:
    """
    Container for duration calculation results.

    Attributes:
        macaulay_duration: Weighted average time to receive cash flows (years).
        modified_duration: Price sensitivity measure (% change per 1% yield change).
        dollar_duration: Dollar value change per 1% yield change (DV01 * 100).
        convexity: Second-order price sensitivity measure.
    """
    macaulay_duration: float
    modified_duration: float
    dollar_duration: float
    convexity: float


@dataclass(frozen=True)
class KeyRateDurationResult:
    """
    Container for key rate duration results.

    Key rate durations decompose interest rate risk across the yield curve,
    showing exposure at specific tenors. This is critical for:
    - Hedging against non-parallel yield curve shifts
    - Understanding barbell vs. bullet portfolio exposures
    - Regulatory reporting (e.g., Basel III IRRBB)

    Attributes:
        krd_2y: Sensitivity to 2-year rate changes.
        krd_5y: Sensitivity to 5-year rate changes.
        krd_10y: Sensitivity to 10-year rate changes.
        krd_30y: Sensitivity to 30-year rate changes.
        total_krd: Sum of all key rate durations (should approximate modified duration).
    """
    krd_2y: float
    krd_5y: float
    krd_10y: float
    krd_30y: float
    total_krd: float


class DurationCalculatorError(Exception):
    """Base exception for duration calculation errors."""
    pass


class InvalidCashFlowError(DurationCalculatorError):
    """Raised when cash flows are invalid."""
    pass


class InvalidYieldError(DurationCalculatorError):
    """Raised when yield is invalid."""
    pass


class InvalidTimeError(DurationCalculatorError):
    """Raised when time values are invalid."""
    pass


class DurationCalculator:
    """
    Calculator for fixed income duration metrics.

    This class provides methods for calculating various duration measures
    used in ALM and fixed income portfolio management. All calculations
    assume discrete cash flows at specified times.

    Attributes:
        frequency: Default compounding frequency for calculations.
        bump_size: Basis point bump for numerical duration calculations (default: 1bp).

    Example:
        >>> calc = DurationCalculator()
        >>> # 4-year bond, 5% coupon, annual payments, $1000 face
        >>> cash_flows = [50, 50, 50, 1050]
        >>> times = [1, 2, 3, 4]
        >>> result = calc.calculate_all_durations(cash_flows, times, ytm=0.05)
        >>> print(f"Modified Duration: {result.modified_duration:.4f}")
    """

    # Standard key rate tenors used in ALM
    KEY_RATE_TENORS = (2.0, 5.0, 10.0, 30.0)

    def __init__(
        self,
        frequency: CompoundingFrequency = CompoundingFrequency.ANNUAL,
        bump_size: float = 0.0001  # 1 basis point
    ) -> None:
        """
        Initialize the duration calculator.

        Args:
            frequency: Default compounding frequency for yield calculations.
            bump_size: Size of yield bump for numerical calculations (default: 1bp).

        Raises:
            ValueError: If bump_size is not positive.
        """
        if bump_size <= 0:
            raise ValueError("bump_size must be positive")

        self.frequency = frequency
        self.bump_size = bump_size

    def _validate_inputs(
        self,
        cash_flows: NDArray[np.float64],
        times: NDArray[np.float64],
        ytm: float
    ) -> None:
        """
        Validate inputs for duration calculations.

        Args:
            cash_flows: Array of cash flow amounts.
            times: Array of times (in years) when cash flows occur.
            ytm: Yield to maturity (as decimal, e.g., 0.05 for 5%).

        Raises:
            InvalidCashFlowError: If cash flows are empty or all zeros.
            InvalidTimeError: If times are invalid or mismatched with cash flows.
            InvalidYieldError: If yield is invalid.
        """
        if len(cash_flows) == 0:
            raise InvalidCashFlowError("Cash flows array cannot be empty")

        if len(cash_flows) != len(times):
            raise InvalidTimeError(
                f"Cash flows length ({len(cash_flows)}) must match times length ({len(times)})"
            )

        if np.all(cash_flows == 0):
            raise InvalidCashFlowError("At least one cash flow must be non-zero")

        if np.any(times < 0):
            raise InvalidTimeError("Times cannot be negative")

        if not np.all(np.diff(times) > 0):
            raise InvalidTimeError("Times must be strictly increasing")

        if ytm <= -1:
            raise InvalidYieldError("Yield must be greater than -100%")

    def _present_value(
        self,
        cash_flows: NDArray[np.float64],
        times: NDArray[np.float64],
        ytm: float
    ) -> float:
        """
        Calculate present value of cash flows.

        Uses discrete discounting: PV = sum(CF_t / (1 + y)^t)

        Args:
            cash_flows: Array of cash flow amounts.
            times: Array of times (in years) when cash flows occur.
            ytm: Yield to maturity (as decimal).

        Returns:
            Present value of all cash flows.
        """
        discount_factors = (1 + ytm) ** (-times)
        return float(np.sum(cash_flows * discount_factors))

    def macaulay_duration(
        self,
        cash_flows: list[float] | NDArray[np.float64],
        times: list[float] | NDArray[np.float64],
        ytm: float
    ) -> float:
        """
        Calculate Macaulay duration.

        Macaulay duration is the weighted average time to receive the bond's
        cash flows, where weights are the present values of each cash flow
        divided by the bond's total present value.

        Formula:
            D_mac = sum(t * PV(CF_t)) / sum(PV(CF_t))

        where:
            - t = time in years
            - PV(CF_t) = present value of cash flow at time t

        Financial Interpretation:
            - For a zero-coupon bond, Macaulay duration equals time to maturity
            - For coupon bonds, it's less than maturity due to interim payments
            - Represents the "center of gravity" of cash flows in time
            - Used in immunization strategies to match asset/liability durations

        Args:
            cash_flows: Cash flow amounts (e.g., [50, 50, 1050] for a bond).
            times: Times in years when cash flows occur (e.g., [1, 2, 3]).
            ytm: Yield to maturity as decimal (e.g., 0.05 for 5%).

        Returns:
            Macaulay duration in years.

        Raises:
            InvalidCashFlowError: If cash flows are invalid.
            InvalidTimeError: If times are invalid.
            InvalidYieldError: If yield is invalid.

        Example:
            >>> calc = DurationCalculator()
            >>> # 3-year bond, 5% annual coupon, 5% yield
            >>> mac_dur = calc.macaulay_duration(
            ...     cash_flows=[50, 50, 1050],
            ...     times=[1, 2, 3],
            ...     ytm=0.05
            ... )
            >>> print(f"{mac_dur:.4f}")  # Approximately 2.86 years
        """
        cf_array = np.asarray(cash_flows, dtype=np.float64)
        t_array = np.asarray(times, dtype=np.float64)

        self._validate_inputs(cf_array, t_array, ytm)

        discount_factors = (1 + ytm) ** (-t_array)
        pv_cash_flows = cf_array * discount_factors
        total_pv = np.sum(pv_cash_flows)

        if total_pv == 0:
            raise InvalidCashFlowError("Total present value is zero")

        weighted_times = t_array * pv_cash_flows
        return float(np.sum(weighted_times) / total_pv)

    def modified_duration(
        self,
        cash_flows: list[float] | NDArray[np.float64],
        times: list[float] | NDArray[np.float64],
        ytm: float,
        frequency: Optional[CompoundingFrequency] = None
    ) -> float:
        """
        Calculate modified duration.

        Modified duration measures the percentage change in bond price for
        a 1% (100 basis point) change in yield. It's derived from Macaulay
        duration by adjusting for the compounding frequency.

        Formula:
            D_mod = D_mac / (1 + y/m)

        where:
            - D_mac = Macaulay duration
            - y = yield to maturity
            - m = compounding frequency per year

        Financial Interpretation:
            - Approximates %ΔP ≈ -D_mod × Δy (for small yield changes)
            - Higher modified duration = greater price sensitivity
            - Critical metric for interest rate risk management
            - Used to calculate DV01 (dollar value of 1 basis point)

        Args:
            cash_flows: Cash flow amounts.
            times: Times in years when cash flows occur.
            ytm: Yield to maturity as decimal.
            frequency: Compounding frequency (defaults to calculator's frequency).

        Returns:
            Modified duration (dimensionless).

        Raises:
            InvalidCashFlowError: If cash flows are invalid.
            InvalidTimeError: If times are invalid.
            InvalidYieldError: If yield is invalid.

        Example:
            >>> calc = DurationCalculator(frequency=CompoundingFrequency.SEMI_ANNUAL)
            >>> mod_dur = calc.modified_duration(
            ...     cash_flows=[25, 25, 25, 25, 25, 1025],
            ...     times=[0.5, 1, 1.5, 2, 2.5, 3],
            ...     ytm=0.05
            ... )
        """
        freq = frequency or self.frequency
        mac_dur = self.macaulay_duration(cash_flows, times, ytm)

        if freq == CompoundingFrequency.CONTINUOUS:
            # For continuous compounding, modified = macaulay
            return mac_dur

        periods_per_year = freq.value
        return mac_dur / (1 + ytm / periods_per_year)

    def convexity(
        self,
        cash_flows: list[float] | NDArray[np.float64],
        times: list[float] | NDArray[np.float64],
        ytm: float
    ) -> float:
        """
        Calculate convexity.

        Convexity measures the curvature of the price-yield relationship,
        providing a second-order correction to duration-based price estimates.

        Formula:
            C = (1/P) × sum(t × (t+1) × PV(CF_t)) / (1+y)²

        Financial Interpretation:
            - Duration gives linear approximation; convexity adds curvature
            - Positive convexity benefits bondholders (price rises more than
              duration predicts when yields fall, falls less when yields rise)
            - More accurate price change: ΔP/P ≈ -D_mod×Δy + 0.5×C×(Δy)²
            - Important for large yield changes and long-dated securities

        Args:
            cash_flows: Cash flow amounts.
            times: Times in years when cash flows occur.
            ytm: Yield to maturity as decimal.

        Returns:
            Convexity (years squared).

        Example:
            >>> calc = DurationCalculator()
            >>> conv = calc.convexity([50, 50, 1050], [1, 2, 3], 0.05)
        """
        cf_array = np.asarray(cash_flows, dtype=np.float64)
        t_array = np.asarray(times, dtype=np.float64)

        self._validate_inputs(cf_array, t_array, ytm)

        discount_factors = (1 + ytm) ** (-t_array)
        pv_cash_flows = cf_array * discount_factors
        total_pv = np.sum(pv_cash_flows)

        if total_pv == 0:
            raise InvalidCashFlowError("Total present value is zero")

        # Convexity formula: sum(t*(t+1)*PV_t) / (P * (1+y)^2)
        weighted_terms = t_array * (t_array + 1) * pv_cash_flows
        return float(np.sum(weighted_terms) / (total_pv * (1 + ytm) ** 2))

    def dollar_duration(
        self,
        cash_flows: list[float] | NDArray[np.float64],
        times: list[float] | NDArray[np.float64],
        ytm: float,
        frequency: Optional[CompoundingFrequency] = None
    ) -> float:
        """
        Calculate dollar duration (DV01 × 100).

        Dollar duration measures the absolute dollar change in bond price
        for a 1% (100bp) change in yield, rather than a percentage change.

        Formula:
            DD = Modified Duration × Price / 100

        Financial Interpretation:
            - Used for hedging and P&L attribution
            - DV01 (dollar value of 1bp) = Dollar Duration / 100
            - Essential for duration-matched hedging strategies

        Args:
            cash_flows: Cash flow amounts.
            times: Times in years when cash flows occur.
            ytm: Yield to maturity as decimal.
            frequency: Compounding frequency.

        Returns:
            Dollar duration (dollar change per 1% yield move).
        """
        cf_array = np.asarray(cash_flows, dtype=np.float64)
        t_array = np.asarray(times, dtype=np.float64)

        price = self._present_value(cf_array, t_array, ytm)
        mod_dur = self.modified_duration(cash_flows, times, ytm, frequency)

        return mod_dur * price / 100

    def calculate_all_durations(
        self,
        cash_flows: list[float] | NDArray[np.float64],
        times: list[float] | NDArray[np.float64],
        ytm: float,
        frequency: Optional[CompoundingFrequency] = None
    ) -> DurationResult:
        """
        Calculate all duration metrics in a single call.

        This is the recommended method for comprehensive duration analysis,
        returning Macaulay duration, modified duration, dollar duration,
        and convexity in a single result object.

        Args:
            cash_flows: Cash flow amounts.
            times: Times in years when cash flows occur.
            ytm: Yield to maturity as decimal.
            frequency: Compounding frequency.

        Returns:
            DurationResult containing all duration metrics.

        Example:
            >>> calc = DurationCalculator()
            >>> result = calc.calculate_all_durations(
            ...     cash_flows=[50, 50, 50, 1050],
            ...     times=[1, 2, 3, 4],
            ...     ytm=0.05
            ... )
            >>> print(f"Macaulay: {result.macaulay_duration:.4f}")
            >>> print(f"Modified: {result.modified_duration:.4f}")
            >>> print(f"Convexity: {result.convexity:.4f}")
        """
        mac_dur = self.macaulay_duration(cash_flows, times, ytm)
        mod_dur = self.modified_duration(cash_flows, times, ytm, frequency)
        dollar_dur = self.dollar_duration(cash_flows, times, ytm, frequency)
        conv = self.convexity(cash_flows, times, ytm)

        return DurationResult(
            macaulay_duration=mac_dur,
            modified_duration=mod_dur,
            dollar_duration=dollar_dur,
            convexity=conv
        )

    def key_rate_durations(
        self,
        cash_flows: list[float] | NDArray[np.float64],
        times: list[float] | NDArray[np.float64],
        spot_rates: dict[float, float],
        tenors: tuple[float, ...] = KEY_RATE_TENORS
    ) -> KeyRateDurationResult:
        """
        Calculate key rate durations at specified tenors.

        Key rate duration (KRD) measures the sensitivity of a bond's price
        to changes in yields at specific points on the yield curve, while
        holding other points constant. This is essential for understanding
        exposure to non-parallel yield curve shifts.

        Methodology:
            1. For each key rate tenor, bump the spot rate at that tenor
            2. Interpolate to create a bumped yield curve
            3. Reprice the bond with the bumped curve
            4. KRD = -(P_up - P_down) / (2 × P × bump_size)

        Financial Interpretation:
            - Sum of KRDs ≈ Modified Duration (for parallel shifts)
            - Identifies "hot spots" on the yield curve
            - Critical for hedging with multiple instruments
            - Required for regulatory reporting (IRRBB under Basel III)

        Yield Curve Interpolation:
            Uses linear interpolation between key rate tenors. For times
            beyond the last tenor, the rate is held constant (flat extrapolation).

        Args:
            cash_flows: Cash flow amounts.
            times: Times in years when cash flows occur.
            spot_rates: Dictionary mapping tenor (years) to spot rate (decimal).
                        Must include rates for each key rate tenor.
                        Example: {0.25: 0.02, 1: 0.025, 2: 0.03, 5: 0.035, 10: 0.04, 30: 0.045}
            tenors: Tuple of key rate tenors to calculate (default: 2y, 5y, 10y, 30y).

        Returns:
            KeyRateDurationResult containing KRD at each tenor.

        Raises:
            ValueError: If spot_rates doesn't contain all required tenors.

        Example:
            >>> calc = DurationCalculator()
            >>> spot_rates = {
            ...     0.5: 0.02, 1: 0.025, 2: 0.03, 3: 0.032,
            ...     5: 0.035, 7: 0.038, 10: 0.04, 20: 0.042, 30: 0.045
            ... }
            >>> krd = calc.key_rate_durations(
            ...     cash_flows=[50, 50, 50, 50, 1050],
            ...     times=[1, 2, 3, 4, 5],
            ...     spot_rates=spot_rates
            ... )
            >>> print(f"5Y KRD: {krd.krd_5y:.4f}")
        """
        cf_array = np.asarray(cash_flows, dtype=np.float64)
        t_array = np.asarray(times, dtype=np.float64)

        # Validate that we have rates for all tenors
        for tenor in tenors:
            if tenor not in spot_rates:
                raise ValueError(f"spot_rates must include rate for {tenor}Y tenor")

        # Sort spot rates by tenor for interpolation
        sorted_tenors = sorted(spot_rates.keys())
        sorted_rates = [spot_rates[t] for t in sorted_tenors]

        def interpolate_rate(t: float) -> float:
            """Linear interpolation of spot rates."""
            if t <= sorted_tenors[0]:
                return sorted_rates[0]
            if t >= sorted_tenors[-1]:
                return sorted_rates[-1]
            return float(np.interp(t, sorted_tenors, sorted_rates))

        def price_with_curve(rate_adjustments: dict[float, float]) -> float:
            """Price bond using adjusted spot curve."""
            total_pv = 0.0
            for cf, t in zip(cf_array, t_array):
                # Get base rate through interpolation
                base_rate = interpolate_rate(t)

                # Apply adjustment based on key rate bucket
                adjustment = 0.0
                for kr_tenor, adj in rate_adjustments.items():
                    # Weight adjustment based on proximity to key rate
                    weight = self._key_rate_weight(t, kr_tenor, tenors)
                    adjustment += weight * adj

                adjusted_rate = base_rate + adjustment
                discount_factor = (1 + adjusted_rate) ** (-t)
                total_pv += cf * discount_factor

            return total_pv

        # Calculate base price
        base_price = price_with_curve({})

        if base_price <= 0:
            raise InvalidCashFlowError("Base price must be positive for KRD calculation")

        # Calculate KRD for each tenor
        krds = {}
        for tenor in tenors:
            # Bump up
            price_up = price_with_curve({tenor: self.bump_size})
            # Bump down
            price_down = price_with_curve({tenor: -self.bump_size})

            # Central difference approximation
            krd = -(price_up - price_down) / (2 * base_price * self.bump_size)
            krds[tenor] = krd

        return KeyRateDurationResult(
            krd_2y=krds.get(2.0, 0.0),
            krd_5y=krds.get(5.0, 0.0),
            krd_10y=krds.get(10.0, 0.0),
            krd_30y=krds.get(30.0, 0.0),
            total_krd=sum(krds.values())
        )

    def _key_rate_weight(
        self,
        t: float,
        key_tenor: float,
        all_tenors: tuple[float, ...]
    ) -> float:
        """
        Calculate the weight of a key rate for a given time.

        Uses triangular weighting where the weight is 1 at the key rate
        tenor and linearly decreases to 0 at adjacent key rate tenors.

        Args:
            t: Time point to calculate weight for.
            key_tenor: The key rate tenor being bumped.
            all_tenors: All key rate tenors in the analysis.

        Returns:
            Weight between 0 and 1.
        """
        sorted_tenors = sorted(all_tenors)
        idx = sorted_tenors.index(key_tenor)

        # Find adjacent tenors
        prev_tenor = sorted_tenors[idx - 1] if idx > 0 else 0.0
        next_tenor = sorted_tenors[idx + 1] if idx < len(sorted_tenors) - 1 else float('inf')

        if t <= prev_tenor:
            return 0.0
        elif t >= next_tenor:
            return 0.0
        elif t <= key_tenor:
            # Rising edge of triangle
            if key_tenor == prev_tenor:
                return 1.0
            return (t - prev_tenor) / (key_tenor - prev_tenor)
        else:
            # Falling edge of triangle
            if next_tenor == float('inf'):
                return 1.0  # Flat extrapolation for last tenor
            return (next_tenor - t) / (next_tenor - key_tenor)
