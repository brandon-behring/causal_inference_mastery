# Advanced Methods

Specialized causal inference methods for complex identification problems.

## Marginal Treatment Effects (MTE)

The MTE framework unifies LATE, ATE, ATT, and policy-relevant treatment effects:

$$MTE(x, u_D) = E[Y_1 - Y_0 | X=x, U_D=u_D]$$

where $U_D$ captures unobserved resistance to treatment. LATE, ATE, and ATT are weighted averages of the MTE.

See {doc}`/api/mte` for API reference.

## Quantile Treatment Effects (QTE)

Estimate treatment effects at different quantiles of the outcome distribution:

$$QTE(\tau) = Q_{Y(1)}(\tau) - Q_{Y(0)}(\tau)$$

Useful when treatment affects the distribution shape, not just the mean.

See {doc}`/api/qte` for API reference.

## Mediation Analysis

Decompose the total effect into direct and indirect (mediated) effects:

- **NDE** (Natural Direct Effect): Effect of $T$ on $Y$ not through mediator $M$
- **NIE** (Natural Indirect Effect): Effect of $T$ on $Y$ through $M$
- **Total effect** = NDE + NIE

See {doc}`/api/mediation` for API reference.

## Control Function

Alternative to IV for nonlinear models (probit, logit):

1. First stage: estimate treatment residuals
2. Second stage: include residuals as control for endogeneity

See {doc}`/api/control-function` for API reference.

## Shift-Share IV (Bartik Instruments)

For settings where national shocks differentially affect regions based on pre-existing industry composition:

$$\Delta Y_l = \beta \sum_k s_{kl} g_k + \epsilon_l$$

Includes Rotemberg diagnostic weights for transparency.

See {doc}`/api/shift-share` for API reference.

## Bunching Estimation

Estimate behavioral responses from excess mass at kink points (e.g., tax brackets):

$$b = \frac{\text{excess mass at kink}}{\text{counterfactual density}}$$

See {doc}`/api/bunching` for API reference.

## Principal Stratification

Estimate effects within latent strata defined by potential intermediate outcomes. Common in noncompliance settings.

See {doc}`/api/principal-strat` for API reference.

## Bounds

When point identification fails, bound the treatment effect:
- **Manski bounds** — Worst-case bounds without assumptions
- **Lee bounds** — Tighter bounds under monotonicity

See {doc}`/api/bounds` for API reference.
