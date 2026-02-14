# Asset-Liability Management (ALM) System

Institutional-grade fixed income analytics for buy-side portfolio management.

## Overview

A comprehensive Python system for bond portfolio management providing:

- **Duration Analytics**: Macaulay, Modified, Key Rate Durations, Convexity, DV01
- **Yield Curve Modeling**: Nelson-Siegel, bootstrapping, spot/forward rates
- **Scenario Analysis**: Parallel shifts, steepening, flattening, inversions
- **Risk Measurement**: Interest rate sensitivity across the curve

Built for buy-side portfolio managers, risk analysts, and quantitative researchers.

## Key Features

### Duration Calculator
- Multiple duration measures for interest rate risk
- Key rate durations at 2y, 5y, 10y, 30y tenors
- Convexity for accurate price sensitivity
- DV01 for dollar-based risk measurement

### Yield Curve Framework
- Construct curves from spot rates or par yields
- Nelson-Siegel parametric fitting
- Forward rate calculations
- Multiple interpolation methods

### Scenario Generation
- Parallel shifts for Fed policy scenarios
- Steepening/flattening for curve positioning
- Custom spread and twist scenarios
- Inversion detection for recession signals

## Installation
```bash
# Clone and setup
git clone https://github.com/YOUR-USERNAME/alm-management.git
cd alm-management

# Create environment
micromamba create -n alm-management python=3.11
micromamba activate alm-management

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v  # 165 tests should pass
```

## Quick Start

## Analysis Notebooks

Jupyter notebooks demonstrating the various calculations:

### Notebook 1: Duration Analysis Demo
**Location:** `notebooks/01_duration_analysis_demo.ipynb`

Analyzes a synthetic 5-bond portfolio (2Y, 5Y, 10Y, 30Y Treasury + 5Y Corporate) calculating:
- Macaulay Duration (weighted average time to cash flows)
- Modified Duration (price sensitivity per 1% yield change)
- Convexity (second-order price sensitivity)
- DV01 (dollar value of 1 basis point move)

**Key Insights:**
- 5Y Corporate bond has lower duration (4.32) than 5Y Treasury (4.49) due to higher coupon
- Duration increases with maturity but at decreasing rate
- Convexity provides cushion against adverse rate moves

**Visualizations:** Bar charts, scatter plots, styled summary tables

---

### Notebook 2: Yield Curve Scenario Analysis
**Location:** `notebooks/02_yield_curve_scenarios.ipynb`

Real-time analysis of U.S. Treasury yield curves using FRED market data:
- Fetches current yields across 11 maturities (1M through 30Y)
- Constructs interpolated yield curve
- Calculates forward rates (1y1y, 5y5y, 10y10y) revealing market expectations
- Fits Nelson-Siegel parametric model
- Runs 5 interest rate scenarios:
  - Current market conditions
  - +50bp parallel shift (Fed tightening)
  - -50bp parallel shift (Fed easing)
  - Steepening scenario (economic expansion)
  - Flattening scenario (recession fears)

**Current Market Insights (as of latest data):**
- Curve shape: STEEP (62bp 2s10s spread)
- Market expects rates to rise (5y5y forward: 4.51% vs 10Y spot: 4.09%)
- Normal upward-sloping curve indicates healthy economic expectations

**Visualizations:** Yield curve plots, scenario overlays, spread analysis

---

### Notebook 3: Integrated Portfolio Analysis *(Coming Next)*
**Location:** `notebooks/03_integrated_analysis.ipynb` *(Planned)*

Combines duration analytics with yield curve scenarios for complete portfolio risk assessment:
- Apply scenarios to bond portfolio from Notebook 1
- Calculate P&L impacts using duration and convexity
- Demonstrate hedging strategies (duration matching, key rate hedging)
- Show barbell vs bullet positioning analysis


### Duration Analysis
```python
from models.duration.duration_calculator import DurationCalculator

calc = DurationCalculator()

# 5-year bond: $1000 face, 5% coupon, annual payments
cash_flows = [50, 50, 50, 50, 1050]
times = [1.0, 2.0, 3.0, 4.0, 5.0]

result = calc.calculate_all_durations(
    cash_flows=cash_flows,
    times=times,
    ytm=0.05
)

print(f"Modified Duration: {result.modified_duration:.4f}")
print(f"Convexity: {result.convexity:.4f}")
print(f"DV01: ${result.dollar_duration/100:.4f}")
```

### Yield Curve & Scenarios
```python
from models.interest_rate.yield_curve import YieldCurve

# Build curve from market data
tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
rates = [0.052, 0.051, 0.0495, 0.047, 0.0455, 0.044, 0.0435, 0.043, 0.0435, 0.044]
curve = YieldCurve.from_spot_rates(tenors=tenors, rates=rates)

# Run scenarios
parallel_up = curve.parallel_shift(50)  # +50bp
steepened = curve.steepen(pivot_tenor=5, amount=25)

print(f"Original 10Y: {curve.spot_rate(10):.4%}")
print(f"Shifted 10Y: {parallel_up.spot_rate(10):.4%}")
```

## Project Structure
```
alm-management/
├── models/
│   ├── duration/
│   │   └── duration_calculator.py    # Duration metrics
│   └── interest_rate/
│       └── yield_curve.py             # Yield curve models
├── tests/
│   └── unit/                          # 165 unit tests
├── docs/
│   └── logs/
│       └── PROJECT_LOG.md             # Development log
├── notebooks/                         # Analysis notebooks
├── requirements.txt
└── README.md
```

## Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific modules
pytest tests/unit/test_duration_calculator.py -v
pytest tests/unit/test_yield_curve.py -v

# Coverage report
pytest tests/ --cov=models --cov-report=html
```

## Financial Concepts

**Macaulay Duration**: Weighted average time to cash flows (years)  
**Modified Duration**: Price sensitivity per 1% yield change  
**Key Rate Duration**: Sensitivity to specific curve points  
**Convexity**: Second-order price sensitivity (curvature)  
**DV01**: Dollar change per 1 basis point move  

**Nelson-Siegel**: 4-parameter yield curve model (β₀, β₁, β₂, τ)  
**Forward Rates**: Implied future rates from spot curve  
**Bootstrapping**: Extract zero rates from coupon bonds  

## Development Roadmap

**Phase 1** ✅ Duration Models (Complete)
- Macaulay, Modified, Key Rate Durations
- Convexity and DV01
- 39 unit tests

**Phase 2** ✅ Yield Curve Framework (Complete)
- Spot/forward rate calculations
- Nelson-Siegel fitting
- Scenario analysis
- 102 unit tests

**Phase 3** (Next) Cash Flow Projection Engine
- Scenario-based projections
- Waterfall analysis

**Phase 4** Liquidity Analysis Dashboard
- Gap analysis
- Interactive visualizations

**Phase 5** Regulatory Reporting
- Basel III IRRBB
- LCR/NSFR calculations

## Technology

- **NumPy**: Numerical computing
- **SciPy**: Optimization algorithms
- **Pytest**: Testing framework
- **Python 3.11**: Type hints and modern syntax

## Documentation

See `docs/logs/PROJECT_LOG.md` for detailed development history and architectural decisions.
