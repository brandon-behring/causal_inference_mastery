# Glossary

```{glossary}
ATE (Average Treatment Effect)
    The expected difference in outcomes between treated and untreated: $E[Y(1) - Y(0)]$.

ATT (Average Treatment Effect on the Treated)
    ATE conditional on being in the treatment group: $E[Y(1) - Y(0) | T=1]$.

Backdoor Criterion
    A set of variables $Z$ satisfies the backdoor criterion if it blocks all backdoor paths from treatment to outcome.

Bootstrap
    Resampling method for constructing confidence intervals and performing hypothesis tests without distributional assumptions.

CATE (Conditional Average Treatment Effect)
    Treatment effect conditional on covariates: $\tau(x) = E[Y(1) - Y(0) | X=x]$.

Conditional Independence
    $Y(0), Y(1) \perp T | X$. Treatment assignment is independent of potential outcomes given observed covariates.

Confounding
    When a variable affects both treatment and outcome, creating a spurious association.

CTE (Conditional Tail Expectation)
    Average of the worst $X\%$ of outcomes. Used in actuarial reserve calculations.

DAG (Directed Acyclic Graph)
    A graphical representation of causal relationships without cycles.

DID (Difference-in-Differences)
    Method comparing changes over time between treated and control groups.

DML (Double Machine Learning)
    Semiparametric method using cross-fitting to avoid regularization bias in CATE estimation.

Doubly Robust
    Estimator that is consistent if either the propensity model or outcome model is correctly specified.

E-value
    Minimum strength of association an unmeasured confounder would need with both treatment and outcome to explain away the observed effect.

Exclusion Restriction
    The assumption that an instrument affects the outcome only through the treatment.

First-Stage F-Statistic
    Test of instrument relevance. F > 10 (Stock-Yogo rule) indicates non-weak instruments.

GMM (Generalized Method of Moments)
    Estimation method that uses moment conditions; efficient with heteroskedasticity.

Granger Causality
    $X$ Granger-causes $Y$ if past values of $X$ improve prediction of $Y$ beyond $Y$'s own history.

Heckman Selection
    Two-step procedure correcting for sample selection bias using an exclusion restriction.

Identification
    The ability to recover a causal parameter from observable data under stated assumptions.

IPW (Inverse Probability Weighting)
    Reweighting observations by the inverse of their treatment probability to remove confounding.

IRF (Impulse Response Function)
    Dynamic effect of a one-unit structural shock traced over time.

IV (Instrumental Variable)
    A variable that affects treatment but has no direct effect on the outcome.

LATE (Local Average Treatment Effect)
    Treatment effect for compliers — those whose treatment status changes with the instrument.

LIML (Limited Information Maximum Likelihood)
    IV estimator more robust to weak instruments than 2SLS, with less finite-sample bias.

MTE (Marginal Treatment Effect)
    Treatment effect at the margin of indifference: $MTE(x, u) = E[Y(1) - Y(0) | X=x, U_D=u]$.

NDE (Natural Direct Effect)
    Effect of treatment on outcome not mediated through $M$.

NIE (Natural Indirect Effect)
    Effect of treatment on outcome operating through mediator $M$.

Parallel Trends
    DiD assumption: treated and control groups would have followed the same trend absent treatment.

Propensity Score
    $e(x) = P(T=1|X=x)$. Probability of treatment given covariates.

RDD (Regression Discontinuity Design)
    Exploits a cutoff in a running variable that determines treatment assignment.

Rosenbaum Bounds
    Sensitivity analysis quantifying how much hidden bias ($\Gamma$) would alter a matched study's conclusions.

SCM (Synthetic Control Method)
    Constructs a weighted combination of donor units to approximate the treated unit's counterfactual.

SUTVA (Stable Unit Treatment Value Assumption)
    No interference between units and one version of treatment.

SVAR (Structural VAR)
    VAR with identifying restrictions to recover causal structural shocks.

TMLE (Targeted Maximum Likelihood Estimation)
    Doubly robust, semiparametric efficient estimator using targeted updates.

2SLS (Two-Stage Least Squares)
    Standard IV estimator. First stage: regress treatment on instruments. Second stage: regress outcome on predicted treatment.

VAR (Vector Autoregression)
    Multivariate time series model where each variable is regressed on lagged values of all variables.

VECM (Vector Error Correction Model)
    VAR for cointegrated non-stationary series with long-run equilibrium constraints.
```
