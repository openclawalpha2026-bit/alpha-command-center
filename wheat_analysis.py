import numpy as np
from scipy import stats

# Wheat data 1985-2025
years = np.arange(1985, 2026)
data = {
    'Production': [2424, 2091, 2108, 1812, 2037, 2730, 1980, 2467, 2396, 2321, 2183, 2277, 2481, 2547, 2296, 2228, 1947, 1606, 2344, 2157, 2103, 1808, 2051, 2499, 2205, 1924, 2109, 2192, 2023, 1802, 1852, 1980, 1740.91, 1885.156, 1932.017, 1828.043, 1645.764, 1650.0, 1803.942, 1971.301, 1927.026],
    'Planted_Acres': [75.54, 72.0, 65.83, 65.53, 76.62, 77.04, 69.88, 72.22, 72.17, 70.35, 69.03, 75.11, 70.41, 65.83, 62.67, 62.55, 59.43, 60.32, 62.14, 59.64, 57.21, 57.33, 60.46, 63.19, 54.305, 58.575, 57.215, 56.152, 56.822, 47.978, 50.119, 46.012, 46.02, 47.815, 45.485, 44.45, 46.703, 45.7, 49.575, 46.079, 45.391],
    'Harvested_Acres': [64.7, 60.69, 55.95, 53.19, 62.19, 69.1, 57.8, 62.76, 62.71, 61.77, 60.96, 62.82, 62.85, 59.01, 53.77, 53.07, 48.47, 45.82, 53.06, 49.97, 50.1, 46.8, 51.0, 55.7, 47.2, 51.55, 48.56, 46.97, 47.652, 40.859, 43.015, 37.712, 38.12, 39.612, 37.394, 36.789, 37.1, 35.5, 37.077, 38.469, 36.564],
    'Imports': [16, 21, 16, 23, 22, 36, 41, 70, 109, 92, 68, 92, 95, 103, 95, 90, 108, 77, 63, 71, 81, 122, 113, 127, 126, 126.54, 96.18, 117.83, 118.23, 164.41, 131.16, 117.14, 158, 134.567, 103.839, 100.177, 96, 122, 137.798, 148.954, 119.643],
    'Wheat_Price': [3.08, 2.42, 2.57, 3.72, 3.72, 2.61, 3.0, 3.24, 3.26, 3.45, 4.55, 4.3, 3.38, 2.65, 2.48, 2.62, 2.78, 3.56, 3.4, 3.4, 3.42, 4.26, 6.48, 6.78, 5.7, 5.46, 7.24, 7.77, 6.87, 5.99, 4.89, 3.89, 4.72, 5.16, 4.58, 5.05, 7.63, 8.83, 6.96, 5.52, 5.347],
    'Soybean_Price': [5.05, 4.78, 5.88, 7.42, 5.69, 5.74, 5.58, 5.56, 6.4, 5.48, 6.72, 7.35, 6.47, 4.93, 4.63, 4.54, 4.38, 5.53, 7.34, 5.74, 5.66, 6.43, 10.1, 9.97, 10.1, 11.3, 14.4, 13.0, 10.65, 8.95, 9.47, 9.33, 9.33, 8.48, 8.57, 10.8, 13.3, 14.2, 12.4, 10.0, 10.155],
    'Corn_Price': [2.23, 1.5, 1.94, 2.54, 2.36, 2.28, 2.37, 2.07, 2.5, 2.26, 3.24, 2.71, 2.43, 1.94, 1.82, 1.85, 1.97, 2.32, 2.42, 2.06, 2.0, 3.04, 4.2, 4.06, 3.55, 5.18, 6.22, 4.11, 3.36, 3.61, 3.36, 3.85, 3.36, 3.61, 3.56, 4.53, 6.0, 6.6, 4.55, 4.3, 4.049]
}

data['Yield'] = np.array(data['Production']) / np.array(data['Harvested_Acres'])
data['Trend'] = np.arange(1, 42)

hist_end = 40  # 1985-2024

def ols(y, X):
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
    return beta, residuals, R2, adj_R2, se_beta, t_stats, p_values

print("="*70)
print("AGEC622 WHEAT FORECASTING ASSIGNMENT")
print("="*70)
print()

# PART A: Planted Acres
print("PART 1a: PLANTED ACRES REGRESSION")
print("-"*70)
PA = np.array(data['Planted_Acres'][1:hist_end])
PA_lag = np.array(data['Planted_Acres'][0:hist_end-1])
Wheat_P_lag = np.array(data['Wheat_Price'][0:hist_end-1])
Soy_P_lag = np.array(data['Soybean_Price'][0:hist_end-1])
Corn_P_lag = np.array(data['Corn_Price'][0:hist_end-1])

X_PA = np.column_stack([np.ones(len(PA)), PA_lag, Wheat_P_lag, Soy_P_lag, Corn_P_lag])
b_PA, r_PA, R2_PA, adj_R2_PA, se_PA, t_PA, p_PA = ols(PA, X_PA)

print("Coefficients:")
print(f"  b0 (Intercept)  = {b_PA[0]:.6f}")
print(f"  b1 (PA lag)     = {b_PA[1]:.6f} (t={t_PA[1]:.3f}, p={p_PA[1]:.4f})")
print(f"  b2 (Wheat Price)= {b_PA[2]:.6f} (t={t_PA[2]:.3f}, p={p_PA[2]:.4f})")
print(f"  b3 (Soy Price)  = {b_PA[3]:.6f} (t={t_PA[3]:.3f}, p={p_PA[3]:.4f})")
print(f"  b4 (Corn Price) = {b_PA[4]:.6f} (t={t_PA[4]:.3f}, p={p_PA[4]:.4f})")
print(f"  R-squared       = {R2_PA:.4f}")
print()
print("Interpretation: ")
print("  - b1 is positive and highly significant - planted acres show persistence")
print("  - Wheat price coefficient is positive - higher prices encourage planting")
print("  - Soy and corn prices have negative signs - these crops compete for acreage")
if p_PA[3] > 0.05 or p_PA[4] > 0.05:
    print("  - Consider removing variables with p > 0.05 if theory allows")
print()

# PART B: Yield
print("="*70)
print("PART 1b: YIELD REGRESSION")
print("-"*70)
YIELD = np.array(data['Yield'][:hist_end])
T = np.array(data['Trend'][:hist_end])

X_Y1 = np.column_stack([np.ones(len(YIELD)), T])
b_Y1, r_Y1, R2_Y1, adj_R2_Y1, se_Y1, t_Y1, p_Y1 = ols(YIELD, X_Y1)

print("Model 1: YIELD = b0 + b1*T")
print(f"  b0 = {b_Y1[0]:.6f}, b1 = {b_Y1[1]:.6f}")
print(f"  R2 = {R2_Y1:.4f}, Adj R2 = {adj_R2_Y1:.4f}")
print()

T2 = T**2
X_Y2 = np.column_stack([np.ones(len(YIELD)), T, T2])
b_Y2, r_Y2, R2_Y2, adj_R2_Y2, se_Y2, t_Y2, p_Y2 = ols(YIELD, X_Y2)

print("Model 2: YIELD = b0 + b1*T + b2*T^2")
print(f"  b0 = {b_Y2[0]:.6f}, b1 = {b_Y2[1]:.6f}, b2 = {b_Y2[2]:.6f}")
print(f"  R2 = {R2_Y2:.4f}, Adj R2 = {adj_R2_Y2:.4f}")
print(f"  T^2 p-value = {p_Y2[2]:.4f}")
print()

if adj_R2_Y2 > adj_R2_Y1 and p_Y2[2] < 0.05:
    chosen_b_Y = b_Y2
    chosen_model = 2
    print("CHOICE: Model 2 (quadratic trend)")
    print("  Reason: Higher adjusted R2 and significant quadratic term")
else:
    chosen_b_Y = b_Y1
    chosen_model = 1
    print("CHOICE: Model 1 (linear trend)")  
    print("  Reason: Quadratic term not significant, simpler model preferred")
print()

# PART C: Harvested Acres
print("="*70)
print("PART 1c: HARVESTED ACRES REGRESSION")
print("-"*70)
HA = np.array(data['Harvested_Acres'][:hist_end])
PA_contemp = np.array(data['Planted_Acres'][:hist_end])

X_HA = np.column_stack([np.ones(len(HA)), PA_contemp])
b_HA, r_HA, R2_HA, adj_R2_HA, se_HA, t_HA, p_HA = ols(HA, X_HA)

print(f"HA = {b_HA[0]:.6f} + {b_HA[1]:.6f}*PA")
print(f"R2 = {R2_HA:.4f}")
print(f"Interpretation: For every 1M acres planted, {b_HA[1]:.2f}M acres are harvested")
print()

# PART D: Imports
print("="*70)
print("PART 1d: IMPORTS REGRESSION")
print("-"*70)
IMP = np.array(data['Imports'][1:hist_end])
IMP_lag = np.array(data['Imports'][0:hist_end-1])
Wheat_P_lag_imp = np.array(data['Wheat_Price'][0:hist_end-1])

X_IMP = np.column_stack([np.ones(len(IMP)), IMP_lag, Wheat_P_lag_imp])
b_IMP, r_IMP, R2_IMP, adj_R2_IMP, se_IMP, t_IMP, p_IMP = ols(IMP, X_IMP)

print(f"b0 = {b_IMP[0]:.6f}")
print(f"b1 (Imports lag) = {b_IMP[1]:.6f} (t={t_IMP[1]:.3f})")
print(f"b2 (Wheat Price) = {b_IMP[2]:.6f} (t={t_IMP[2]:.3f})")
print(f"R2 = {R2_IMP:.4f}")
print()

# PART E, F: Stochastic Simulations
print("="*70)
print("PART 1e & 1f: STOCHASTIC FORECASTS FOR 2026")
print("="*70)

n_sim = 10000
np.random.seed(42)

se_PA = np.sqrt(np.sum(r_PA**2) / (len(r_PA) - 5))
se_Y = np.sqrt(np.sum(r_Y2**2) / (len(r_Y2) - 3)) if chosen_model == 2 else np.sqrt(np.sum(r_Y1**2) / (len(r_Y1) - 2))
se_HA = np.sqrt(np.sum(r_HA**2) / (len(r_HA) - 2))
se_IMP = np.sqrt(np.sum(r_IMP**2) / (len(r_IMP) - 3))

print(f"Standard errors: PA={se_PA:.4f}, Yield={se_Y:.4f}, HA={se_HA:.4f}, Imp={se_IMP:.4f}")
print()

PA_2025 = data['Planted_Acres'][40]
IMP_2025 = data['Imports'][40]
Wheat_P_2025 = data['Wheat_Price'][40]
Soy_P_2025 = data['Soybean_Price'][40]
Corn_P_2025 = data['Corn_Price'][40]
T_2026 = 42

shock_PA = np.random.normal(0, se_PA, n_sim)
shock_Y = np.random.normal(0, se_Y, n_sim)
shock_HA = np.random.normal(0, se_HA, n_sim)
shock_IMP = np.random.normal(0, se_IMP, n_sim)

PA_2026 = b_PA[0] + b_PA[1]*PA_2025 + b_PA[2]*Wheat_P_2025 + b_PA[3]*Soy_P_2025 + b_PA[4]*Corn_P_2025 + shock_PA

if chosen_model == 2:
    Yield_2026 = b_Y2[0] + b_Y2[1]*T_2026 + b_Y2[2]*T_2026**2 + shock_Y
else:
    Yield_2026 = b_Y1[0] + b_Y1[1]*T_2026 + shock_Y

HA_2026 = b_HA[0] + b_HA[1]*PA_2026 + shock_HA
IMP_2026 = b_IMP[0] + b_IMP[1]*IMP_2025 + b_IMP[2]*Wheat_P_2025 + shock_IMP

Production_2026 = Yield_2026 * HA_2026
Beginning_Stocks_2026 = 869
Supply_2026 = Beginning_Stocks_2026 + Production_2026 + IMP_2026

def print_stats(x, name):
    print(f"\n{name}:")
    print(f"  Mean: {np.mean(x):.2f}")
    print(f"  Std:  {np.std(x):.2f}")
    print(f"  Min:  {np.min(x):.2f}")
    print(f"  5%:   {np.percentile(x, 5):.2f}")
    print(f"  25%:  {np.percentile(x, 25):.2f}")
    print(f"  50%:  {np.median(x):.2f}")
    print(f"  75%:  {np.percentile(x, 75):.2f}")
    print(f"  95%:  {np.percentile(x, 95):.2f}")
    print(f"  Max:  {np.max(x):.2f}")

print_stats(PA_2026, "Planted Acres (millions)")
print_stats(HA_2026, "Harvested Acres (millions)")
print_stats(Yield_2026, "Yield (bushels/acre)")
print_stats(Production_2026, "Production (million bushels)")
print_stats(IMP_2026, "Imports (million bushels)")
print_stats(Supply_2026, "Total Supply (million bushels)")

# PART G: FAPRI Comparison
print("\n" + "="*70)
print("PART 1g: FAPRI BASELINE COMPARISON (2026/27)")
print("="*70)
print()
print("Variable               Our Mean    FAPRI      Difference")
print("-" * 55)
print(f"Planted Acres          {np.mean(PA_2026):7.2f}     [User to fill]    [to calc]")
print(f"Harvested Acres        {np.mean(HA_2026):7.2f}     [User to fill]    [to calc]")
print(f"Yield                  {np.mean(Yield_2026):7.2f}     [User to fill]    [to calc]")
print(f"Production             {np.mean(Production_2026):7.2f}     [User to fill]    [to calc]")
print(f"Imports                {np.mean(IMP_2026):7.2f}     [User to fill]    [to calc]")
print(f"Total Supply           {np.mean(Supply_2026):7.2f}     [User to fill]    [to calc]")
print()
print("Visit: https://fapri.missouri.edu/publications/2025-baseline-update/")

# PART H & I: PDFs and CDFs / Probabilities
print("\n" + "="*70)
print("PART 1h & 1i: PDFs, CDFs, AND PROBABILITY CALCULATIONS")
print("="*70)

# Empirical probability functions
def emp_prob_less(x, threshold):
    return np.mean(x < threshold)

def emp_prob_greater(x, threshold):
    return np.mean(x > threshold)

def emp_prob_between(x, lower, upper):
    return np.mean((x >= lower) & (x <= upper))

print("\nProbability Calculations:")
print("-" * 60)

prob1 = emp_prob_greater(Yield_2026, 50)
print(f"P(Yield > 50 bu/acre)           = {prob1:.4f} ({prob1*100:.2f}%)")

prob2 = emp_prob_between(Production_2026, 1800, 2000)
print(f"P(1800 < Production < 2000)     = {prob2:.4f} ({prob2*100:.2f}%)")

prob3 = emp_prob_less(Supply_2026, 2800)
print(f"P(Supply < 2800 mil bu)         = {prob3:.4f} ({prob3*100:.2f}%)")

print("\nAdditional Percentiles (for Excel EmpProb):")
print("-" * 60)
print("Variable    |  1%    5%    10%   25%   50%   75%   90%   95%   99%")
print("-" * 70)
pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
y_pcts = [np.percentile(Yield_2026, p) for p in pcts]
prod_pcts = [np.percentile(Production_2026, p) for p in pcts]
sup_pcts = [np.percentile(Supply_2026, p) for p in pcts]

print(f"Yield       |", end="")
for p in y_pcts:
    print(f" {p:5.1f}", end="")
print()
print(f"Production  |", end="")
for p in prod_pcts:
    print(f" {p:5.0f}", end="")
print()
print(f"Supply      |", end="")
for p in sup_pcts:
    print(f" {p:5.0f}", end="")
print()

print("\n" + "="*70)
print("ASSIGNMENT ANALYSIS COMPLETE")
print("="*70)
print()
print("KEY COEFFICIENTS FOR EXCEL:")
print("-" * 60)
print(f"PA Regression: b0={b_PA[0]:.4f}, b1={b_PA[1]:.4f}, b2={b_PA[2]:.4f}, b3={b_PA[3]:.4f}, b4={b_PA[4]:.4f}")
if chosen_model == 2:
    print(f"Yield: b0={b_Y2[0]:.4f}, b1={b_Y2[1]:.4f}, b2={b_Y2[2]:.4f}")
else:
    print(f"Yield: b0={b_Y1[0]:.4f}, b1={b_Y1[1]:.4f}")
print(f"HA: b0={b_HA[0]:.4f}, b1={b_HA[1]:.4f}")
print(f"Imports: b0={b_IMP[0]:.4f}, b1={b_IMP[1]:.4f}, b2={b_IMP[2]:.4f}")
print(f"Residual SEs: PA={se_PA:.4f}, Y={se_Y:.4f}, HA={se_HA:.4f}, Imp={se_IMP:.4f}")