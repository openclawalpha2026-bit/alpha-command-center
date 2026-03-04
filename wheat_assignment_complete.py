"""
AGEC622 Wheat Forecasting Assignment - Complete Solution
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# ===================== DATA PREPARATION =====================
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

# Calculate YIELD
data['Yield'] = np.array(data['Production']) / np.array(data['Harvested_Acres'])
data['Trend'] = np.arange(1, len(data['Year']) + 1)

hist_end = 40  # Index for 2024

print("=" * 70)
print("AGEC622 WHEAT FORECASTING ASSIGNMENT - COMPLETE SOLUTION")
print("=" * 70)
print()

# OLS regression function
def ols_regression(y, X):
    n = len(y)
    k = X.shape[1]
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat
    SST = np.sum((y - np.mean(y))**2)
    SSE = np.sum(residuals**2)
    R2 = 1 - SSE/SST
    adj_R2 = 1 - (SSE/(n-k))/(SST/(n-1))
    sigma2 = SSE / (n - k)
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))
    return beta, residuals, R2, adj_R2, se_beta, t_stats, p_values, y_hat

# ===================== PART A: PLANTED ACRES =====================
print("=" * 70)
print("PART 1a: WHEAT PLANTED ACRES (PA) REGRESSION")
print("=" * 70)
print()
print("Model: PA_t = beta0 + beta1*PA_{t-1} + beta2*Wheat_P_{t-1} + beta3*Soy_P_{t-1} + beta4*Corn_P_{t-1}")
print()

PA = np.array(data['Planted_Acres'][1:hist_end])
PA_lag = np.array(data['Planted_Acres'][0:hist_end-1])
Wheat_P_lag = np.array(data['Wheat_Price'][0:hist_end-1])
Soy_P_lag = np.array(data['Soybean_Price'][0:hist_end-1])
Corn_P_lag = np.array(data['Corn_Price'][0:hist_end-1])

X_PA = np.column_stack([np.ones(len(PA)), PA_lag, Wheat_P_lag, Soy_P_lag, Corn_P_lag])
beta_PA, resid_PA, R2_PA, adj_R2_PA, se_PA, t_PA, p_PA, y_hat_PA = ols_regression(PA, X_PA)

print(f"Sample: 1986-2024 (n = {len(PA)})")
print()
print("REGRESSION RESULTS:")
print("-" * 60)
print(f"{'Parameter':<25} {'Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Intercept (b0)':<25} {beta_PA[0]:>12.6f} {se_PA[0]:>10.6f} {t_PA[0]:>10.3f} {p_PA[0]:>10.4f}")
print(f"{'PA_lag (b1)':<25} {beta_PA[1]:>12.6f} {se_PA[1]:>10.6f} {t_PA[1]:>10.3f} {p_PA[1]:>10.4f}")
print(f"{'Wheat Price_lag (b2)':<25} {beta_PA[2]:>12.6f} {se_PA[2]:>10.6f} {t_PA[2]:>10.3f} {p_PA[2]:>10.4f}")
print(f"{'Soy Price_lag (b3)':<25} {beta_PA[3]:>12.6f} {se_PA[3]:>10.6f} {t_PA[3]:>10.3f} {p_PA[3]:>10.4f}")
print(f"{'Corn Price_lag (b4)':<25} {beta_PA[4]:>12.6f} {se_PA[4]:>10.6f} {t_PA[4]:>10.3f} {p_PA[4]:>10.4f}")
print("-" * 60)
print(f"{'R-squared':<25} {R2_PA:>12.4f}")
print(f"{'Adjusted R-squared':<25} {adj_R2_PA:>12.4f}")
print()

print("INTERPRETATION OF COEFFICIENTS:")
print(f"  b1 (lagged PA) = {beta_PA[1]:.4f}: Positive - strong persistence in planted acres")
print(f"  b2 (wheat price) = {beta_PA[2]:.4f}: {'Positive - higher prices encourage planting' if beta_PA[2] > 0 else 'Negative'}")
print(f"  b3 (soy price) = {beta_PA[3]:.4f}: {'Negative - soy competes for acreage' if beta_PA[3] < 0 else 'Positive - unexpected'}")
print(f"  b4 (corn price) = {beta_PA[4]:.4f}: {'Negative - corn competes for acreage' if beta_PA[4] < 0 else 'Positive - unexpected'}")
print()

insignificant = []
if p_PA[2] > 0.05: insignificant.append(f"Wheat Price (p={p_PA[2]:.4f})")
if p_PA[3] > 0.05: insignificant.append(f"Soybean Price (p={p_PA[3]:.4f})")
if p_PA[4] > 0.05: insignificant.append(f"Corn Price (p={p_PA[4]:.4f})")

if insignificant:
    print("VARIABLES TO CONSIDER REMOVING (p > 0.05):")
    for var in insignificant:
        print(f"  * {var}")
else:
    print("All price variables are statistically significant (p < 0.05).")
print()

# ===================== PART B: YIELD REGRESSION =====================
print("=" * 70)
print("PART 1b: WHEAT YIELD REGRESSION")
print("=" * 70)
print()
print(f"Yield = Production / Harvested Acres")
print(f"Mean yield (1985-2024): {np.mean(data['Yield'][:hist_end]):.4f} bushels/acre")
print()

YIELD = np.array(data['Yield'][:hist_end])
T = np.array(data['Trend'][:hist_end])

X_yield1 = np.column_stack([np.ones(len(YIELD)), T])
beta_Y1, resid_Y1, R2_Y1, adj_R2_Y1, se_Y1, t_Y1, p_Y1, y_hat_Y1 = ols_regression(YIELD, X_yield1)

print("MODEL 1: Linear Trend")
print("-" * 60)
print(f"{'Parameter':<20} {'Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Intercept':<20} {beta_Y1[0]:>12.6f} {se_Y1[0]:>10.6f} {t_Y1[0]:>10.3f} {p_Y1[0]:>10.4f}")
print(f"{'Trend (T)':<20} {beta_Y1[1]:>12.6f} {se_Y1[1]:>10.6f} {t_Y1[1]:>10.3f} {p_Y1[1]:>10.4f}")
print(f"{'R-squared':<20} {R2_Y1:>12.4f}")
print()

T2 = T**2
X_yield2 = np.column_stack([np.ones(len(YIELD)), T, T2])
beta_Y2, resid_Y2, R2_Y2, adj_R2_Y2, se_Y2, t_Y2, p_Y2, y_hat_Y2 = ols_regression(YIELD, X_yield2)

print("MODEL 2: Quadratic Trend")
print("-" * 60)
print(f"{'Parameter':<20} {'Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Intercept':<20} {beta_Y2[0]:>12.6f} {se_Y2[0]:>10.6f} {t_Y2[0]:>10.3f} {p_Y2[0]:>10.4f}")
print(f"{'Trend (T)':<20} {beta_Y2[1]:>12.6f} {se_Y2[1]:>10.6f} {t_Y2[1]:>10.3f} {p_Y2[1]:>10.4f}")
print(f"{'Trend^2 (T^2)':<20} {beta_Y2[2]:>12.6f} {se_Y2[2]:>10.6f} {t_Y2[2]:>10.3f} {p_Y2[2]:>10.4f}")
print(f"{'R-squared':<20} {R2_Y2:>12.4f}")
print(f"{'Adjusted R-squared':<20} {adj_R2_Y2:>12.4f}")
print()

print("MODEL SELECTION:")
if adj_R2_Y2 > adj_R2_Y1 and p_Y2[2] < 0.05:
    chosen_beta_Y = beta_Y2
    chosen_resid_Y = resid_Y2
    chosen_model_Y = 2
    print(f"  SELECTED: Model 2 (Quadratic) - Adj R2={adj_R2_Y2:.4f} > {adj_R2_Y1:.4f}, T^2 significant (p={p_Y2[2]:.4f})")
else:
    chosen_beta_Y = beta_Y1
    chosen_resid_Y = resid_Y1
    chosen_model_Y = 1
    print(f"  SELECTED: Model 1 (Linear) - T^2 not significant (p={p_Y2[2]:.4f})")
print()

# ===================== PART C: HARVESTED ACRES =====================
print("=" * 70)
print("PART 1c: HARVESTED ACRES REGRESSION")
print("=" * 70)
print()
print("Model: HA_t = beta0 + beta1*PA_t")
print()

HA = np.array(data['Harvested_Acres'][:hist_end])
PA_contemp = np.array(data['Planted_Acres'][:hist_end])
X_HA = np.column_stack([np.ones(len(HA)), PA_contemp])
beta_HA, resid_HA, R2_HA, adj_R2_HA, se_HA_coef, t_HA, p_HA, y_hat_HA = ols_regression(HA, X_HA)

print(f"Sample: 1985-2024 (n = {len(HA)})")
print()
print("REGRESSION RESULTS:")
print("-" * 60)
print(f"{'Parameter':<25} {'Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Intercept (b0)':<25} {beta_HA[0]:>12.6f} {se_HA_coef[0]:>10.6f} {t_HA[0]:>10.3f} {p_HA[0]:>10.4f}")
print(f"{'Planted Acres (b1)':<25} {beta_HA[1]:>12.6f} {se_HA_coef[1]:>10.6f} {t_HA[1]:>10.3f} {p_HA[1]:>10.4f}")
print("-" * 60)
print(f"{'R-squared':<25} {R2_HA:>12.4f}")
print(f"{'Adjusted R-squared':<25} {adj_R2_HA:>12.4f}")
print()
print(f"INTERPRETATION: For every 1M additional acres planted, {beta_HA[1]:.4f}M acres are harvested.")
print()

# ===================== PART D: IMPORTS =====================
print("=" * 70)
print("PART 1d: WHEAT IMPORTS REGRESSION")
print("=" * 70)
print()
print("Model: Imports_t = beta0 + beta1*Imports_{t-1} + beta2*Wheat_P_{t-1}")
print()

IMP = np.array(data['Imports'][1:hist_end])
IMP_lag = np.array(data['Imports'][0:hist_end-1])
Wheat_P_lag_imp = np.array(data['Wheat_Price'][0:hist_end-1])
X_IMP = np.column_stack([np.ones(len(IMP)), IMP_lag, Wheat_P_lag_imp])
beta_IMP, resid_IMP, R2_IMP, adj_R2_IMP, se_IMP_coef, t_IMP, p_IMP, y_hat_IMP = ols_regression(IMP, X_IMP)

print(f"Sample: 1986-2024 (n = {len(IMP)})")
print()
print("REGRESSION RESULTS:")
print("-" * 60)
print(f"{'Parameter':<25} {'Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Intercept (b0)':<25} {beta_IMP[0]:>12.6f} {se_IMP_coef[0]:>10.6f} {t_IMP[0]:>10.3f} {p_IMP[0]:>10.4f}")
print(f"{'Imports_lag (b1)':<25} {beta_IMP[1]:>12.6f} {se_IMP_coef[1]:>10.6f} {t_IMP[1]:>10.3f} {p_IMP[1]:>10.4f}")
print(f"{'Wheat Price_lag (b2)':<25} {beta_IMP[2]:>12.6f} {se_IMP_coef[2]:>10.6f} {t_IMP[2]:>10.3f} {p_IMP[2]:>10.4f}")
print("-" * 60)
print(f"{'R-squared':<25} {R2_IMP:>12.4f}")
print(f"{'Adjusted R-squared':<25} {adj_R2_IMP:>12.4f}")
print()

# ===================== PARTS E & F: STOCHASTIC SIMULATION =====================
print("=" * 70)
print("PARTS 1e & 1f: STOCHASTIC SIMULATION FOR 2026 (n=10,000 draws)")
print("=" * 70)
print()

n_sim = 10000
np.random.seed(42)

# Residual standard errors (for simulation shocks)
rse_PA  = np.sqrt(np.sum(resid_PA**2)  / (len(resid_PA)  - 5))
rse_Y   = np.sqrt(np.sum(chosen_resid_Y**2) / (len(chosen_resid_Y) - (3 if chosen_model_Y == 2 else 2)))
rse_HA  = np.sqrt(np.sum(resid_HA**2)  / (len(resid_HA)  - 2))
rse_IMP = np.sqrt(np.sum(resid_IMP**2) / (len(resid_IMP) - 3))

print(f"Residual SEs: PA={rse_PA:.4f}, Yield={rse_Y:.4f}, HA={rse_HA:.4f}, Imports={rse_IMP:.4f}")
print()

PA_2025      = data['Planted_Acres'][40]
IMP_2025     = data['Imports'][40]
Wheat_P_2025 = data['Wheat_Price'][40]
Soy_P_2025   = data['Soybean_Price'][40]
Corn_P_2025  = data['Corn_Price'][40]
T_2026       = 42

shock_PA  = np.random.normal(0, rse_PA,  n_sim)
shock_Y   = np.random.normal(0, rse_Y,   n_sim)
shock_HA  = np.random.normal(0, rse_HA,  n_sim)
shock_IMP = np.random.normal(0, rse_IMP, n_sim)

PA_2026 = (beta_PA[0] + beta_PA[1]*PA_2025 + beta_PA[2]*Wheat_P_2025
           + beta_PA[3]*Soy_P_2025 + beta_PA[4]*Corn_P_2025 + shock_PA)

if chosen_model_Y == 2:
    Yield_2026 = chosen_beta_Y[0] + chosen_beta_Y[1]*T_2026 + chosen_beta_Y[2]*T_2026**2 + shock_Y
else:
    Yield_2026 = chosen_beta_Y[0] + chosen_beta_Y[1]*T_2026 + shock_Y

HA_2026  = beta_HA[0]  + beta_HA[1]*PA_2026 + shock_HA
IMP_2026 = beta_IMP[0] + beta_IMP[1]*IMP_2025 + beta_IMP[2]*Wheat_P_2025 + shock_IMP

Production_2026       = Yield_2026 * HA_2026
Beginning_Stocks_2026 = 869.0
Supply_2026           = Beginning_Stocks_2026 + Production_2026 + IMP_2026

def print_dist(arr, label):
    print(f"  {label}:")
    print(f"    Mean   : {np.mean(arr):>10.3f}")
    print(f"    Std    : {np.std(arr):>10.3f}")
    print(f"    5th %  : {np.percentile(arr,  5):>10.3f}")
    print(f"    25th % : {np.percentile(arr, 25):>10.3f}")
    print(f"    Median : {np.median(arr):>10.3f}")
    print(f"    75th % : {np.percentile(arr, 75):>10.3f}")
    print(f"    95th % : {np.percentile(arr, 95):>10.3f}")
    print()

print("SIMULATION RESULTS (2026 Forecasts):")
print()
print_dist(PA_2026,         "Planted Acres (mil)")
print_dist(HA_2026,         "Harvested Acres (mil)")
print_dist(Yield_2026,      "Yield (bu/acre)")
print_dist(Production_2026, "Production (mil bu)")
print_dist(IMP_2026,        "Imports (mil bu)")
print_dist(Supply_2026,     "Total Supply (mil bu)")

# ===================== PART G: FAPRI COMPARISON =====================
print("=" * 70)
print("PART 1g: COMPARISON WITH FAPRI BASELINE")
print("=" * 70)
print()
fapri_PA    = 48.0
fapri_HA    = 39.0
fapri_yield = 49.5
fapri_prod  = 1930.0
fapri_imp   = 120.0

mean_PA   = np.mean(PA_2026)
mean_HA   = np.mean(HA_2026)
mean_Y    = np.mean(Yield_2026)
mean_prod = np.mean(Production_2026)
mean_imp  = np.mean(IMP_2026)

print(f"{'Variable':<30} {'Our Mean':>15} {'FAPRI 26/27':>15} {'Difference':>12}")
print("-" * 75)
print(f"{'Planted Acres (mil)':<30} {mean_PA:>15.2f} {fapri_PA:>15.1f} {mean_PA - fapri_PA:>12.2f}")
print(f"{'Harvested Acres (mil)':<30} {mean_HA:>15.2f} {fapri_HA:>15.1f} {mean_HA - fapri_HA:>12.2f}")
print(f"{'Yield (bu/acre)':<30} {mean_Y:>15.2f} {fapri_yield:>15.1f} {mean_Y - fapri_yield:>12.2f}")
print(f"{'Production (mil bu)':<30} {mean_prod:>15.2f} {fapri_prod:>15.1f} {mean_prod - fapri_prod:>12.2f}")
print(f"{'Imports (mil bu)':<30} {mean_imp:>15.2f} {fapri_imp:>15.1f} {mean_imp - fapri_imp:>12.2f}")
print()
print("Note: FAPRI values are approximate. Verify at https://fapri.missouri.edu/")
print()

# ===================== PARTS H & I: PDFs / CDFs / PROBABILITIES =====================
print("=" * 70)
print("PARTS 1h & 1i: PDFs, CDFs, AND PROBABILITY CALCULATIONS")
print("=" * 70)
print()

prob_yield_gt50    = np.mean(Yield_2026 > 50)
prob_prod_range    = np.mean((Production_2026 >= 1800) & (Production_2026 <= 2000))
prob_supply_lt2800 = np.mean(Supply_2026 < 2800)

print("Sample Probability Calculations:")
print(f"  P(Yield > 50 bu/acre)              = {prob_yield_gt50:.4f}  ({prob_yield_gt50*100:.2f}%)")
print(f"  P(1800 < Production < 2000 mil bu) = {prob_prod_range:.4f}  ({prob_prod_range*100:.2f}%)")
print(f"  P(Supply < 2800 mil bu)            = {prob_supply_lt2800:.4f}  ({prob_supply_lt2800*100:.2f}%)")
print()

pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("Percentile Table:")
header = f"{'Variable':<12}" + "".join(f" {p:>6}%" for p in pcts)
print(header)
print("-" * len(header))
for arr, name in [(Yield_2026, "Yield"), (Production_2026, "Prod"), (Supply_2026, "Supply")]:
    row = f"{name:<12}" + "".join(f" {np.percentile(arr, p):>7.1f}" for p in pcts)
    print(row)
print()

print("=" * 70)
print("KEY COEFFICIENTS SUMMARY (for Excel entry)")
print("=" * 70)
print()
print(f"PA:      b0={beta_PA[0]:.4f}, b1={beta_PA[1]:.4f}, b2={beta_PA[2]:.4f}, b3={beta_PA[3]:.4f}, b4={beta_PA[4]:.4f}")
if chosen_model_Y == 2:
    print(f"Yield:   b0={chosen_beta_Y[0]:.4f}, b1={chosen_beta_Y[1]:.4f}, b2={chosen_beta_Y[2]:.4f}  (Model 2 - Quadratic)")
else:
    print(f"Yield:   b0={chosen_beta_Y[0]:.4f}, b1={chosen_beta_Y[1]:.4f}  (Model 1 - Linear)")
print(f"HA:      b0={beta_HA[0]:.4f}, b1={beta_HA[1]:.4f}")
print(f"Imports: b0={beta_IMP[0]:.4f}, b1={beta_IMP[1]:.4f}, b2={beta_IMP[2]:.4f}")
print(f"Residual SEs: PA={rse_PA:.4f}, Yield={rse_Y:.4f}, HA={rse_HA:.4f}, Imports={rse_IMP:.4f}")
print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
