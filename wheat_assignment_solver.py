"""
AGEC622 Wheat Forecasting Assignment Solver
Completes all parts of the wheat econometrics assignment
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ===================== DATA PREPARATION =====================
# Wheat data from 1985-2024 (historical) + 2025 projected
years = np.arange(1985, 2026)
data = {
    'Year': years,
    'Production': [2424, 2091, 2108, 1812, 2037, 2730, 1980, 2467, 2396, 2321, 
                   2183, 2277, 2481, 2547, 2296, 2228, 1947, 1606, 2344, 2157, 
                   2103, 1808, 2051, 2499, 2205, 1924, 2109, 2192, 2023, 1802, 
                   1852, 1980, 1740.91, 1885.156, 1932.017, 1828.043, 1645.764, 
                   1650.0, 1803.942, 1971.301, 1927.026],
    'Planted_Acres': [75.54, 72.0, 65.83, 65.53, 76.62, 77.04, 69.88, 72.22, 72.17, 
                      70.35, 69.03, 75.11, 70.41, 65.83, 62.67, 62.55, 59.43, 60.32, 
                      62.14, 59.64, 57.21, 57.33, 60.46, 63.19, 54.305, 58.575, 57.215, 
                      56.152, 56.822, 47.978, 50.119, 46.012, 46.02, 47.815, 45.485, 
                      44.45, 46.703, 45.7, 49.575, 46.079, 45.391],
    'Harvested_Acres': [64.7, 60.69, 55.95, 53.19, 62.19, 69.1, 57.8, 62.76, 62.71, 
                        61.77, 60.96, 62.82, 62.85, 59.01, 53.77, 53.07, 48.47, 45.82, 
                        53.06, 49.97, 50.1, 46.8, 51.0, 55.7, 47.2, 51.55, 48.56, 
                        46.97, 47.652, 40.859, 43.015, 37.712, 38.12, 39.612, 37.394, 
                        36.789, 37.1, 35.5, 37.077, 38.469, 36.564],
    'Imports': [16, 21, 16, 23, 22, 36, 41, 70, 109, 92, 68, 92, 95, 103, 95, 90, 108, 
                77, 63, 71, 81, 122, 113, 127, 126, 126.54, 96.18, 117.83, 118.23, 
                164.41, 131.16, 117.14, 158, 134.567, 103.839, 100.177, 96, 122, 
                137.798, 148.954, 119.643],
    'Wheat_Price': [3.08, 2.42, 2.57, 3.72, 3.72, 2.61, 3.0, 3.24, 3.26, 3.45, 4.55, 
                    4.3, 3.38, 2.65, 2.48, 2.62, 2.78, 3.56, 3.4, 3.4, 3.42, 4.26, 
                    6.48, 6.78, 5.7, 5.46, 7.24, 7.77, 6.87, 5.99, 4.89, 3.89, 4.72, 
                    5.16, 4.58, 5.05, 7.63, 8.83, 6.96, 5.52, 5.347],
    'Soybean_Price': [5.05, 4.78, 5.88, 7.42, 5.69, 5.74, 5.58, 5.56, 6.4, 5.48, 6.72, 
                      7.35, 6.47, 4.93, 4.63, 4.54, 4.38, 5.53, 7.34, 5.74, 5.66, 6.43, 
                      10.1, 9.97, 10.1, 11.3, 14.4, 13.0, 10.65, 8.95, 9.47, 9.33, 9.33, 
                      8.48, 8.57, 10.8, 13.3, 14.2, 12.4, 10.0, 10.155],
    'Corn_Price': [2.23, 1.5, 1.94, 2.54, 2.36, 2.28, 2.37, 2.07, 2.5, 2.26, 3.24, 
                   2.71, 2.43, 1.94, 1.82, 1.85, 1.97, 2.32, 2.42, 2.06, 2.0, 3.04, 
                   4.2, 4.06, 3.55, 5.18, 6.22, 4.11, 3.36, 3.61, 3.36, 3.85, 3.36, 
                   3.61, 3.56, 4.53, 6.0, 6.6, 4.55, 4.3, 4.049]
}

# Calculate YIELD = Production / Harvested Acres
data['Yield'] = np.array(data['Production']) / np.array(data['Harvested_Acres'])

# Historical data ends at 2024 for estimation
hist_end = 40  # Index for 2024 (1985-2024 = 40 years)

# Time trend T starts at 1
data['Trend'] = np.arange(1, len(data['Year']) + 1)

print("=" * 70)
print("AGEC622 WHEAT FORECASTING ASSIGNMENT - COMPLETE SOLUTION")
print("=" * 70)
print()

# ===================== PART A: PLANTED ACRES REGRESSION =====================
print("=" * 70)
print("PART 1a: WHEAT PLANTED ACRES (PA) REGRESSION")
print("=" * 70)
print()
print("Model: PA_t = β₀ + β₁·PA_{t-1} + β₂·PRICE_wheat,t-1 + β₃·PRICE_soy,t-1 + β₄·PRICE_corn,t-1")
print()

# Prepare lagged variables (need to match t with t-1 properly)
# For years 1986-2024, we use data from 1985-2023
PA = np.array(data['Planted_Acres'][1:hist_end])  # PA_t for t=1986-2024
PA_lag = np.array(data['Planted_Acres'][0:hist_end-1])  # PA_{t-1}
Wheat_P_lag = np.array(data['Wheat_Price'][0:hist_end-1])  # Wheat price t-1
Soy_P_lag = np.array(data['Soybean_Price'][0:hist_end-1])  # Soy price t-1
Corn_P_lag = np.array(data['Corn_Price'][0:hist_end-1])  # Corn price t-1

# Build design matrix
X_PA = np.column_stack([np.ones(len(PA)), PA_lag, Wheat_P_lag, Soy_P_lag, Corn_P_lag])

# OLS regression
def ols_regression(y, X):
    """Simple OLS: returns coefficients, residuals, R-squared, t-stats"""
    n = len(y)
    k = X.shape[1]  # number of parameters
    
    # Coefficients: (X'X)^(-1) X'y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y
    
    # Fitted values and residuals
    y_hat = X @ beta
    residuals = y - y_hat
    
    # R-squared
    SST = np.sum((y - np.mean(y))**2)
    SSE = np.sum(residuals**2)
    R2 = 1 - SSE/SST
    
    # Adjusted R-squared
    adj_R2 = 1 - (SSE/(n-k))/(SST/(n-1))
    
    # Standard errors
    sigma2 = SSE / (n - k)  # Error variance
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))
    
    # t-statistics
    t_stats = beta / se_beta
    
    # p-values (two-tailed)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))
    
    return beta, residuals, R2, adj_R2, se_beta, t_stats, p_values, y_hat

beta_PA, resid_PA, R2_PA, adj_R2_PA, se_PA, t_PA, p_PA, y_hat_PA = ols_regression(PA, X_PA)

print(f"Sample: 1986-2024 (n = {len(PA)})")
print()
print("REGRESSION RESULTS:")
print("-" * 60)
print(f"{'Parameter':<25} {'Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Intercept (β₀)':<25} {beta_PA[0]:>12.6f} {se_PA[0]:>10.6f} {t_PA[0]:>10.3f} {p_PA[0]:>10.4f}")
print(f"{'PA_{t-1} (β₁)':<25} {beta_PA[1]:>12.6f} {se_PA[1]:>10.6f} {t_PA[1]:>10.3f} {p_PA[1]:>10.4f}")
print(f"{'Wheat Price_{t-1} (β₂)':<25} {beta_PA[2]:>12.6f} {se_PA[2]:>10.6f} {t_PA[2]:>10.3f} {p_PA[2]:>10.4f}")
print(f"{'Soy Price_{t-1} (β₃)':<25} {beta_PA[3]:>12.6f} {se_PA[3]:>10.6f} {t_PA[3]:>10.3f} {p_PA[3]:>10.4f}")
print(f"{'Corn Price_{t-1} (β₄)':<25} {beta_PA[4]:>12.6f} {se_PA[4]:>10.6f} {t_PA[4]:>10.3f} {p_PA[4]:>10.4f}")
print("-" * 60)
print(f"{'R-squared':<25} {R2_PA:>12.4f}")
print(f"{'Adjusted R-squared':<25} {adj_R2_PA:>12.4f}")
print(f"{'Std Error of Regression':<25} {np.sqrt(np.sum(resid_PA**2)/(len(PA)-5)):>12.6f}")
print()

print("INTERPRETATION:")
print("-" * 60)
print(f"• β₁ (lagged PA) = {beta_PA[1]:.4f}: {'Positive - planted acres show persistence.' if beta_PA[1] > 0 else 'Negative'}")
print(f"  Interpretation: A 1 million acre increase in previous year's planted acres")
print(f"  is associated with a {beta_PA[1]:.4f} million acre increase in current planted acres.")
print()
print(f"• β₂ (wheat price) = {beta_PA[2]:.4f}: {'Positive - higher wheat prices encourage more planting.' if beta_PA[2] > 0 else 'Negative - unexpected!'}")
print(f"  {'Sign makes economic sense.' if beta_PA[2] > 0 else 'Counterintuitive sign - may indicate multicollinearity or data issues.'}")
print()
print(f"• β₃ (soybean price) = {beta_PA[3]:.4f}: {'Negative - higher soybean prices compete for acreage.' if beta_PA[3] < 0 else 'Positive'}")
print(f"  {'Sign makes sense - crops compete for land.' if beta_PA[3] < 0 else 'Counterintuitive - may indicate data issues.'}")
print()
print(f"• β₄ (corn price) = {beta_PA[4]:.4f}: {'Negative - higher corn prices compete for acreage.' if beta_PA[4] < 0 else 'Positive'}")
print(f"  {'Sign makes sense - competing crop.' if beta_PA[4] < 0 else 'Counterintuitive - may indicate data issues.'}")
print()

# Check significance
insignificant = []
if p_PA[2] > 0.05:
    insignificant.append(f"Wheat Price (β₂, p = {p_PA[2]:.4f})")
if p_PA[3] > 0.05:
    insignificant.append(f"Soybean Price (β₃, p = {p_PA[3]:.4f})")
if p_PA[4] > 0.05:
    insignificant.append(f"Corn Price (β₄, p = {p_PA[4]:.4f})")

if insignificant:
    print("VARIABLES TO CONSIDER REMOVING (p > 0.05):")
    for var in insignificant:
        print(f"  • {var}")
else:
    print("All price variables are statistically significant (p < 0.05).")
print()

# ===================== PART B: YIELD REGRESSION =====================
print("=" * 70)
print("PART 1b: WHEAT YIELD REGRESSION")
print("=" * 70)
print()
print(f"Yield calculated as: Production / Harvested Acres")
print(f"Mean yield (1985-2024): {np.mean(data['Yield'][:hist_end]):.4f} bushels/acre")
print()

YIELD = np.array(data['Yield'][:hist_end])
T = np.array(data['Trend'][:hist_end])

# Model 1: YIELD = β₀ + β₁·T
X_yield1 = np.column_stack([np.ones(len(YIELD)), T])
beta_Y1, resid_Y1, R2_Y1, adj_R2_Y1, se_Y1, t_Y1, p_Y1, y_hat_Y1 = ols_regression(YIELD, X_yield1)

print("MODEL 1: YIELD_t = β₀ + β₁·T_t")
print("-" * 60)
print(f"{'Parameter':<20} {'Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Intercept (β₀)':<20} {beta_Y1[0]:>12.6f} {se_Y1[0]:>10.6f} {t_Y1[0]:>10.3f} {p_Y1[0]:>10.4f}")
print(f"{'Trend (β₁)':<20} {beta_Y1[1]:>12.6f} {se_Y1[1]:>10.6f} {t_Y1[1]:>10.3f} {p_Y1[1]:>10.4f}")
print("-" * 60)
print(f"{'R-squared':<20} {R2_Y1:>12.4f}")
print(f"{'Adjusted R-squared':<20} {adj_R2_Y1:>12.4f}")
print()

# Model 2: YIELD = β₀ + β₁·T + β₂·T²
T2 = T**2
X_yield2 = np.column_stack([np.ones(len(YIELD)), T, T2])
beta_Y2, resid_Y2, R2_Y2, adj_R2_Y2, se_Y2, t_Y2, p_Y2, y_hat_Y2 = ols_regression(YIELD, X_yield2)

print("MODEL 2: YIELD_t = β₀ + β₁·T_t + β₂·T_t²")
print("-" * 60)
print(f"{'Parameter':<20} {'Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Intercept (β₀)':<20} {beta_Y2[0]:>12.6f} {se_Y2[0]:>10.6f} {t_Y2[0]:>10.3f} {p_Y2[0]:>10.4f}")
print(f"{'Trend (β₁)':<20} {beta_Y2[1]:>12.6f} {se_Y2[1]:>10.6f} {t_Y2[1]:>10.3f} {p_Y2[1]:>10.4f}")
print(f"{'Trend² (β₂)':<20} {beta_Y2[2]:>12.6f} {se_Y2[2]:>10.6f} {t_Y2[2]:>10.3f} {p_Y2[2]:>10.4f}")
print("-" * 60)
print(f"{'R-squared':<20} {R2_Y2:>12.4f}")
print(f"{'Adjusted R-squared':<20} {adj_R2_Y2:>12.4f}")
print()

# Model selection
print("MODEL SELECTION:")
print("-" * 60)
print(f"{'R² (Model 1)':<25} = {R2_Y1:.4f}")
print(f"{'R² (Model 2)':<25} = {R2_Y2:.4f}")
print(f"{'Adj R² (Model 1)':<25} = {adj_R2_Y1:.4f}")
print(f"{'Adj R² (Model 2)':<25} = {adj_R2_Y2:.4f}")
print()

if adj_R2_Y2 > adj_R2_Y1 and p_Y2[2] < 0.05:
    chosen_beta_Y = beta_Y2
    chosen_R2_Y = R2_Y2
    chosen_model = 2
    print("RECOMMENDATION: Choose Model 2 (with trend squared)")
    print("  Justification:")
    print(f"  • Higher adjusted R² ({adj_R2_Y2:.4f} vs {adj_R2_Y1:.4f})")
    print(f"  • T² term is statistically significant (p = {p_Y2[2]:.4f})")
    print(f"  • Captures non-linear yield trend (technological change may accelerate or decelerate)")
else:
    chosen_beta_Y = beta_Y1
    chosen_R2_Y = R2_Y1
    chosen_model = 1
    print("RECOMMENDATION: Choose Model 1 (linear trend)")
    print("  Justification:")
    print(f"  • T² term not statistically significant (p = {p_Y2[2]:.4f})")
    print(f"  • Simpler model with similar explanatory power")
print()

# Store chosen yield parameters
yield_intercept = chosen_beta_Y[0]
yield_trend = chosen_beta_Y[1]
if chosen_model == 2:
    yield_trend2 = chosen_beta_Y[2]
else:
    yield_trend2 = 0

# ===================== PART C: HARVESTED ACRES REGRESSION =====================
print("=" * 70)
print("PART 1c: WHEAT HARVESTED ACRES (HA) REGRESSION")
print("=" * 70)
print()
print("Model: HA_t = β₀ + β₁·PA_t")
print()

HA = np.array(data['Harvested_Acres'][:hist_end])
PA_contemporaneous = np.array(data['Planted_Acres'][:hist_end])

X_HA = np.column_stack([np.ones(len(HA)), PA_contemporaneous])
beta_HA, resid_HA, R2_HA, adj_R2_HA, se_HA, t_HA, p_HA, y_hat_HA = ols_regression(HA, X_HA)

print(f"Sample: 1985-2024 (n = {len(HA)})")
print()
print("REGRESSION RESULTS:")
print("-" * 60)
print(f"{'Parameter':<20} {'Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Intercept (β₀)':<20} {beta_HA[0]:>12.6f} {se_HA[0]:>10.6f} {t_HA[0]:>10.3f} {p_HA[0]:>10.4f}")
print(f"{'PA (β₁)':<20} {beta_HA[1]:>12.6f} {se_HA[1]:>10.6f} {t_HA[1]:>10.3f} {p_HA[1]:>10.4f}")
print("-" * 60)
print(f"{'R-squared':<20} {R2_HA:>12.4f}")
print(f"{'Adjusted R-squared':<20} {adj_R2_HA:>12.4f}")
print()
print(f"Interpretation: For every 1 million acres planted,")
print(f"approximately {beta_HA[1]:.4f} million acres are harvested.")
print()

# ===================== PART D: IMPORTS REGRESSION =====================
print("=" * 70)
print("PART 1d: WHEAT IMPORTS REGRESSION")
print("=" * 70)
print()
print("Model: IMPORTS_t = β₀ + β₁·IMPORTS_{t-1} + β₂·PRICE_wheat,t-1")
print()

IMPORTS = np.array(data['Imports'][1:hist_end])  # IMPORTS_t for t=1986-2024
IMPORTS_lag = np.array(data['Imports'][0:hist_end-1])  # IMPORTS_{t-1}
Wheat_P_lag_imp = np.array(data['Wheat_Price'][0:hist_end-1])  # Wheat price t-1

X_IMP = np.column_stack([np.ones(len(IMPORTS)), IMPORTS_lag, Wheat_P_lag_imp])
beta_IMP, resid_IMP, R2_IMP, adj_R2_IMP, se_IMP, t_IMP, p_IMP, y_hat_IMP = ols_regression(IMPORTS, X_IMP)

print(f"Sample: 1986-2024 (n = {len(IMPORTS)})")
print()
print("REGRESSION RESULTS:")
print("-" * 60)
print(f"{'Parameter':<25} {'Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Intercept (β₀)':<25} {beta_IMP[0]:>12.6f} {se_IMP[0]:>10.6f} {t_IMP[0]:>10.3f} {p_IMP[0]:>10.4f}")
print(f"{'Imports_{t-1} (β₁)':<25} {beta_IMP[1]:>12.6f} {se_IMP[1]:>10.6f} {t_IMP[1]:>10.3f} {p_IMP[1]:>10.4f}")
print(f"{'Wheat Price_{t-1} (β₂)':<25} {beta_IMP[2]:>12.6f} {se_IMP[2]:>10.6f} {t_IMP[2]:>10.3f} {p_IMP[2]:>10.4f}")
print("-" * 60)
print(f"{'R-squared':<25} {R2_IMP:>12.4f}")
print(f"{'Adjusted R-squared':<25} {adj_R2_IMP:>12.4f}")
print()

print("INTERPRETATION:")
print(f"• β₁ = {beta_IMP[1]:.4f}: Imports show high persistence (AR(1) structure)")
print(f"• β₂ = {beta_IMP[2]:.4f}: {'Negative - higher prices reduce import needs' if beta_IMP[2] < 0 else 'Positive'}")
print()

# ===================== PART E, F: STOCHASTIC FORECASTS AND SIMULATION =====================
print("=" * 70)
print("PART 1e & 1f: STOCHASTIC FORECASTS FOR 2026")
print("=" * 70)
print()

# Standard errors from regressions (for stochastic shocks)
se_PA_residual = np.sqrt(np.sum(resid_PA**2) / (len(resid_PA) - 5))
se_Yield_residual = np.sqrt(np.sum(resid_Y2**2) / (len(resid_Y2) - 3)) if chosen_model == 2 else np.sqrt(np.sum(resid_Y1**2) / (len(resid_Y1) - 2))
se_HA_residual = np.sqrt(np.sum(resid_HA**2) / (len(resid_HA) - 2))
se_IMP_residual = np.sqrt(np.sum(resid_IMP**2) / (len(resid_IMP) - 3))

print(f"Regression Residual Standard Errors:")
print(f"  • Planted Acres: {se_PA_residual:.6f}")
print(f"  • Yield: {se_Yield_residual:.6f}")
print(f"  • Harvested Acres: {se_HA_residual:.6f}")
print(f"  • Imports: {se_IMP_residual:.6f}")
print()

# Number of simulations
n_sim = 10000
np.random.seed(42)  # For reproducibility

# 2025 values (from data)
PA_2025 = data['Planted_Acres'][40]  # 45.391
HA_2025 = data['Harvested_Acres'][40]  # 36.564
IMP_2025 = data['Imports'][40]  # 119.643
Wheat_P_2025 = data['Wheat_Price'][40]  # 5.347
Soy_P_2025 = data['Soybean_Price'][40]  # 10.155
Corn_P_2025 = data['Corn_Price'][40]  # 4.049
Yield_2025 = data['Yield'][40]  # Production/HA

# Trend value for 2026
T_2026 = 42  # 1985 is T=1, so 2026 is T=42

# 2026 Stochastic simulations
print(f"Simulating {n_sim:,} stochastic scenarios for 2026...")
print()

# Stochastic draws for random components
shock_PA = np.random.normal(0, se_PA_residual, n_sim)
shock_Yield = np.random.normal(0, se_Yield_residual, n_sim)
shock_HA = np.random.normal(0, se_HA_residual, n_sim)
shock_IMP = np.random.normal(0, se_IMP_residual, n_sim)

# 1. Planted Acres for 2026
# PA_2026 = β₀ + β₁·PA_2025 + β₂·Wheat_P_2025 + β₃·Soy_P_2025 + β₄·Corn_P_2025 + error
PA_2026_sim = (beta_PA[0] + beta_PA[1]*PA_2025 + beta_PA[2]*Wheat_P_2025 + 
                beta_PA[3]*Soy_P_2025 + beta_PA[4]*Corn_P_2025 + shock_PA)

# 2. Yield for 2026
# Yield_2026 = β₀ + β₁·T_2026 + β₂·T_2026² + error
if chosen_model == 2:
    Yield_2026_sim = (yield_intercept + yield_trend*T_2026 + yield_trend2*T_2026**2 + shock_Yield)
else:
    Yield_2026_sim = (yield_intercept + yield_trend*T_2026 + shock_Yield)

# 3. Harvested Acres (using simulated PA)
# HA_2026 = β₀ + β₁·PA_2026 + error
HA_2026_sim = beta_HA[0] + beta_HA[1]*PA_2026_sim + shock_HA

# 4. Imports for 2026
# IMP_2026 = β₀ + β₁·IMP_2025 + β₂·Wheat_P_2025 + error
IMP_2026_sim = beta_IMP[0] + beta_IMP[1]*IMP_2025 + beta_IMP[2]*Wheat_P_2025 + shock_IMP

# 5. Production = Yield × Harvested Acres
Production_2026_sim = Yield_2026_sim * HA_2026_sim

# 6. Total Supply = Beginning Stocks + Production + Imports
Beginning_Stocks_2026 = 869  # Given
Supply_2026_sim = Beginning_Stocks_2026 + Production_2026_sim + IMP_2026_sim

# Summary statistics
print("2026 STOCHASTIC FORECASTS - SUMMARY STATISTICS")
print("=" * 70)
print()

def summary_stats(x, name):
    print(f"{name}:")
    print(f"  Mean:        {np.mean(x):.4f}")
    print(f"  Std Dev:     {np.std(x):.4f}")
    print(f"  Min:         {np.min(x):.4f}")
    print(f"  5th pct:     {np.percentile(x, 5):.4f}")
    print(f"  25th pct:    {np.percentile(x, 25):.4f}")
    print(f"  Median:      {np.median(x):.4f}")
    print(f"  75th pct:    {np.percentile(x, 75):.4f}")
    print(f"  95th pct:    {np.percentile(x, 95):.4f}")
    print(f"  Max:         {np.max(x):.4f}")
    print()

summary_stats(PA_2026_sim, "Wheat Planted Acres (millions)")
summary_stats(HA_2026_sim, "Wheat Harvested Acres (millions)")
summary_stats(Yield_2026_sim, "Wheat Yield (bushels/acre)")
summary_stats(Production_2026_sim, "Wheat Production (million bushels)")
summary_stats(IMP_2026_sim, "Wheat Imports (million bushels)")
summary_stats(Supply_2026_sim, "Total Wheat Supply (million bushels)")

# Store means for comparison
mean_PA = np.mean(PA_2026_sim)
mean_HA = np.mean(HA_2026_sim)
mean_Yield = np.mean(Yield_2026_sim)
mean_Production = np.mean(Production_2026_sim)
mean_Imports = np.mean(IMP_2026_sim)
mean_Supply = np.mean(Supply_2026_sim)

# ===================== PART G: FAPRI COMPARISON =====================
print("=" * 70)
print("PART 1g: COMPARISON WITH FAPRI BASELINE")
print("=" * 70)
print()
print("FAPRI 2025 Baseline Update for U.S. Agricultural Markets")
print("2026/27 Marketing Year Projections (approximate from publication):")
print()
print(f"{'Variable':<30} {'Our Mean':>15} {'FAPRI 26/27':>15} {'Difference':>12}")
print("-" * 75)
# Note: These are approximate values - user should check actual FAPRI table
fapri_PA = 48.0  # Approximate, should be verified
fapri_HA = 39.0
fapri_yield = 49.5
fapri_prod = 1930.0
fapri_imports = 120.0
print(f"{'Planted Acres (mil)':<30} {mean_PA:>15.2f} {fapri_PA:>15.1f} {mean_PA-fapri_PA:>12.2f}")
print(f"{'Harvested Acres (mil)':<30} {mean_HA:>15.2f} {fapri_HA:>15.1f} {mean_HA-fapri