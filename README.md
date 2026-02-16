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
git clone https://github.com/WickedoneC/alm-management.git
cd alm-management

# Create environment
micromamba create -n alm-management python=3.11
micromamba activate alm-management

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v  # 102 tests should pass
```

## Quick Start

```python
from models.duration.duration_calculator import DurationCalculator
from models.interest_rate.yield_curve import YieldCurve

# Calculate duration for a 5-year Treasury bond
calc = DurationCalculator()
result = calc.calculate_all_durations(
    cash_flows=[25, 25, 25, 25, 25, 25, 25, 25, 25, 1025],
    times=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    ytm=0.04
)
print(f"Modified Duration: {result.modified_duration:.2f} years")
print(f"Convexity: {result.convexity:.2f}")

# Build a yield curve and run scenarios
curve = YieldCurve.from_spot_rates(
    tenors=[0.25, 0.5, 1, 2, 5, 10, 30],
    rates=[0.045, 0.044, 0.043, 0.041, 0.040, 0.042, 0.045]
)
shocked = curve.parallel_shift(50)  # +50bp rate shock
print(f"10Y spot: {curve.spot_rate(10):.2%} → {shocked.spot_rate(10):.2%}")
```

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

### Notebook 3: Integrated Portfolio Analysis 
**Location:** `notebooks/03_integrated_analysis.ipynb` 

Combines duration analytics with yield curve scenarios for complete portfolio risk assessment:
**Portfolio Composition:**
- 5-bond portfolio: 2Y, 5Y, 10Y, 30Y Treasury + 5Y Corporate
- Total value: ~$5,000
- Weighted average duration: 6.8 years
- Key concentration: 40% in 5Y sector

**Analysis Components:**
1. **Portfolio Summary** - Aggregate duration, convexity, DV01, KRD profile
2. **Scenario P&L Analysis** - Apply 5 yield curve scenarios to calculate portfolio impact
3. **Key Rate Duration Exposure** - Identify concentration risks by maturity
4. **Hedging Strategies** - Duration matching, KRD hedging approaches
5. **Stress Testing** - Extreme scenarios (+200bp shock, inversions)
6. **Recommendations** - Actionable portfolio management insights

**Scenario Results (on $5,028 portfolio):**
- +50bp parallel shock: -$171.65 loss (-3.4%)
- -50bp parallel drop: +$184.52 gain (+3.7%)
- Steepening: -$17.78 loss (vulnerable to curve moves)
- Flattening: +$18.17 gain (benefits from flattening)

**Key Insights:**
- Positive convexity provides $13 asymmetric benefit on ±50bp moves
- 30Y sector has highest sensitivity ($78-87 P&L range)
- 5Y concentration (40%) creates intermediate curve vulnerability
- Portfolio positioned for stable/falling rate environment

**Visualizations:**
- KRD exposure bar chart by maturity bucket
- Scenario P&L with color-coded gains/losses
- P&L contribution heatmap (maturity × scenario)


## Project Structure
```
alm-management/
├── models/
│   ├── duration/
│   │   └── duration_calculator.py     # Duration metrics engine
│   └── interest_rate/
│       └── yield_curve.py              # Yield curve models
├── tests/
│   └── unit/                           # 102 unit tests
│       ├── test_duration_calculator.py  # 39 duration tests
│       └── test_yield_curve.py          # 63 yield curve tests
├── notebooks/
│   ├── 01_duration_analysis_demo.ipynb  # Synthetic portfolio analysis
│   ├── 02_yield_curve_scenarios.ipynb   # FRED data + scenarios
│   └── 03_integrated_analysis.ipynb     # Combined risk assessment
├── docs/
│   └── logs/
│       └── PROJECT_LOG.md              # Development log
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

**Phase 1** ✅ Duration Models
- Macaulay, Modified, Key Rate Durations
- Convexity and DV01
- 39 unit tests

**Phase 2** ✅ Yield Curve Framework
- Spot/forward rate calculations
- Nelson-Siegel fitting
- Scenario analysis
- 63 unit tests (102 total)

**Phase 3** ✅ Integrated Analysis Notebooks
- Duration analysis with synthetic portfolio
- Yield curve scenarios with FRED market data
- Combined portfolio risk assessment

## Technology

- **NumPy**: Numerical computing
- **SciPy**: Optimization algorithms
- **Pytest**: Testing framework
- **Python 3.11**: Type hints and modern syntax

## Documentation

See `docs/logs/PROJECT_LOG.md` for detailed development history and architectural decisions.
