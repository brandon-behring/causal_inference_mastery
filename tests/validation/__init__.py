"""
Validation test suite for causal inference estimators.

This package implements a six-layer validation architecture:
1. Known-answer tests (in tests/test_rct/)
2. Adversarial tests (adversarial/)
3. Monte Carlo validation (monte_carlo/)
4. Python↔Julia cross-validation (cross_language/)
5. R triangulation (future)
6. Golden reference tests (in tests/test_rct/)

See docs/PYTHON_VALIDATION_ARCHITECTURE.md for complete documentation.
"""

__version__ = "1.0.0"
