"""CATE (Conditional Average Treatment Effect) estimation module.

This module implements meta-learners for heterogeneous treatment effect estimation:
- S-Learner: Single model approach
- T-Learner: Two separate models approach
- X-Learner: Cross-learner (planned)
- R-Learner: Robinson transformation (planned)

References
----------
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects
  using machine learning." PNAS 116(10): 4156-4165.
- Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous treatment effects."
  Biometrika 108(2): 299-319.
"""

from .base import CATEResult
from .meta_learners import s_learner, t_learner

__all__ = [
    "CATEResult",
    "s_learner",
    "t_learner",
]
