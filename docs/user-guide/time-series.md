# Time Series Causal Methods

Causal inference in time series settings — Granger causality, structural VARs, impulse responses.

## Methods

### Vector Autoregression (VAR)

Reduced-form VAR for multivariate time series:

$$Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + \cdots + A_p Y_{t-p} + u_t$$

### Structural VAR (SVAR)

Imposes identifying restrictions to recover structural shocks:
- Short-run restrictions (Cholesky decomposition)
- Long-run restrictions (Blanchard-Quah)
- Sign restrictions

### Granger Causality

Tests whether past values of $X$ improve prediction of $Y$ beyond $Y$'s own history:

$$H_0: X \text{ does not Granger-cause } Y$$

### Impulse Response Functions (IRF)

Trace the dynamic effect of a one-standard-deviation structural shock over time.

### VECM & Cointegration

For non-stationary series with long-run equilibrium relationships.

## Example

```python
from causal_inference.timeseries import VAR, granger_causality

# Fit VAR
var = VAR(max_lags=4, ic="aic")
result = var.fit(data[["gdp", "inflation", "interest_rate"]])

# Granger causality test
gc = granger_causality(data, cause="interest_rate", effect="inflation", max_lags=4)
print(f"F-stat: {gc.f_stat:.2f}, p: {gc.p_value:.4f}")

# Impulse responses
irf = result.impulse_response(impulse="interest_rate", response="inflation", periods=20)
```

See {doc}`/api/timeseries` for full API reference.
