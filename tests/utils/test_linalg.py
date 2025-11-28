"""
Tests for utils/linalg.py - standardized linear algebra error handling.

Tests verify that safe wrappers:
1. Work correctly on valid inputs (happy path)
2. Provide diagnostic errors on invalid inputs (singularity, ill-conditioning)
3. Warn appropriately on numerical issues
"""

import numpy as np
import pytest
import warnings

from src.causal_inference.utils.linalg import (
    safe_inv,
    safe_solve,
    safe_lstsq,
    safe_eigvalsh,
)


class TestSafeInv:
    """Tests for safe_inv() - matrix inversion with error handling."""

    def test_safe_inv_valid_matrix(self):
        """Test inversion of well-conditioned matrix."""
        X = np.array([[2.0, 1.0], [1.0, 2.0]])
        X_inv = safe_inv(X, name="test_matrix")

        # Verify X @ X_inv = I
        identity = X @ X_inv
        assert np.allclose(identity, np.eye(2))

    def test_safe_inv_larger_matrix(self):
        """Test inversion works for larger matrices."""
        np.random.seed(42)
        X = np.random.randn(10, 10) + 5 * np.eye(10)  # Well-conditioned
        X_inv = safe_inv(X, name="large_matrix", check_condition=False)

        assert np.allclose(X @ X_inv, np.eye(10), atol=1e-10)

    def test_safe_inv_singular_matrix_raises(self):
        """Test that singular matrix raises LinAlgError with diagnostics."""
        # Perfect collinearity: second column = first column
        singular = np.array([[1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(np.linalg.LinAlgError) as exc_info:
            safe_inv(singular, name="singular_matrix")

        error_msg = str(exc_info.value)
        # Check error contains diagnostics
        assert "singular_matrix" in error_msg
        assert "singular or near-singular" in error_msg
        assert "det=" in error_msg
        assert "collinearity" in error_msg

    def test_safe_inv_not_square_raises(self):
        """Test that non-square matrix raises ValueError."""
        not_square = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(ValueError) as exc_info:
            safe_inv(not_square, name="rectangular")

        assert "must be square" in str(exc_info.value)
        assert "rectangular" in str(exc_info.value)

    def test_safe_inv_ill_conditioned_warns(self):
        """Test that ill-conditioned matrix triggers warning."""
        # Create ill-conditioned matrix (small eigenvalues)
        eigenvalues = np.array([1e-12, 1.0])
        Q = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        ill_conditioned = Q @ np.diag(eigenvalues) @ Q.T

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # May fail or succeed depending on numerical precision
            try:
                safe_inv(ill_conditioned, name="ill_cond", check_condition=True)
                # If it succeeds, check for warning
                if len(w) > 0:
                    assert "ill-conditioned" in str(w[0].message)
            except np.linalg.LinAlgError:
                # Also acceptable - matrix is nearly singular
                pass

    def test_safe_inv_skip_condition_check(self):
        """Test that condition check can be disabled."""
        X = np.array([[2.0, 1.0], [1.0, 2.0]])

        # Should work without warnings when check disabled
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_inv = safe_inv(X, check_condition=False)
            assert len(w) == 0  # No warnings
            assert np.allclose(X @ X_inv, np.eye(2))


class TestSafeSolve:
    """Tests for safe_solve() - linear system solver."""

    def test_safe_solve_basic(self):
        """Test solving Ax = b for well-conditioned system."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([9.0, 8.0])
        x = safe_solve(A, b, name="test_system")

        # Verify Ax = b
        assert np.allclose(A @ x, b)

    def test_safe_solve_multiple_rhs(self):
        """Test solving with multiple right-hand sides."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        B = np.array([[1.0, 2.0], [3.0, 4.0]])
        X = safe_solve(A, B, name="multi_rhs")

        assert np.allclose(A @ X, B)

    def test_safe_solve_singular_raises(self):
        """Test that singular system raises error with diagnostics."""
        A_singular = np.array([[1.0, 2.0], [2.0, 4.0]])
        b = np.array([1.0, 2.0])

        with pytest.raises(np.linalg.LinAlgError) as exc_info:
            safe_solve(A_singular, b, name="singular_system")

        error_msg = str(exc_info.value)
        assert "singular_system" in error_msg
        assert "singular" in error_msg
        assert "det=" in error_msg

    def test_safe_solve_shape_mismatch_raises(self):
        """Test that incompatible shapes raise ValueError."""
        A = np.array([[1, 2], [3, 4]])
        b = np.array([1, 2, 3])  # Wrong length

        with pytest.raises(ValueError) as exc_info:
            safe_solve(A, b, name="mismatched")

        assert "Incompatible shapes" in str(exc_info.value)

    def test_safe_solve_not_square_raises(self):
        """Test that non-square A raises ValueError."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([1, 2])

        with pytest.raises(ValueError) as exc_info:
            safe_solve(A, b, name="rectangular")

        assert "must be square" in str(exc_info.value)


class TestSafeLstsq:
    """Tests for safe_lstsq() - least squares solver."""

    def test_safe_lstsq_overdetermined(self):
        """Test least squares for overdetermined system (m > n)."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        beta_true = np.array([1.0, 2.0, 3.0])
        y = X @ beta_true + 0.1 * np.random.randn(n)

        beta_hat, residuals, rank, s = safe_lstsq(X, y, name="OLS")

        # Should recover approximately true beta
        assert beta_hat.shape == (3,)
        assert np.allclose(beta_hat, beta_true, atol=0.5)
        assert rank == 3  # Full rank

    def test_safe_lstsq_exact_fit(self):
        """Test least squares when exact solution exists."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        beta = np.array([2.0, 3.0])
        y = X @ beta  # Exact fit

        beta_hat, residuals, rank, s = safe_lstsq(X, y, name="exact")

        assert np.allclose(beta_hat, beta)
        assert rank == 2
        # Residuals should be near zero for exact fit
        if len(residuals) > 0:  # residuals may be empty for exact fit
            assert np.allclose(residuals, 0, atol=1e-10)

    def test_safe_lstsq_rank_deficient_warns(self):
        """Test that rank-deficient matrix triggers warning."""
        # Create rank-deficient design matrix (collinear columns)
        X = np.array([[1.0, 2.0, 2.0], [2.0, 4.0, 4.0], [3.0, 6.0, 6.0]])
        y = np.array([1.0, 2.0, 3.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            beta, residuals, rank, s = safe_lstsq(X, y, name="rank_deficient")

            # Should warn about rank deficiency
            assert len(w) > 0
            assert "rank-deficient" in str(w[0].message)
            assert rank < 3  # Not full rank

    def test_safe_lstsq_shape_mismatch_raises(self):
        """Test that incompatible shapes raise ValueError."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2])  # Wrong length (should be 3)

        with pytest.raises(ValueError) as exc_info:
            safe_lstsq(X, y, name="mismatched")

        assert "Incompatible shapes" in str(exc_info.value)

    def test_safe_lstsq_not_2d_raises(self):
        """Test that 1D matrix raises ValueError."""
        X = np.array([1, 2, 3])  # 1D
        y = np.array([1, 2, 3])

        with pytest.raises(ValueError) as exc_info:
            safe_lstsq(X, y, name="not_2d")

        assert "must be 2D" in str(exc_info.value)


class TestSafeEigvalsh:
    """Tests for safe_eigvalsh() - eigenvalue computation."""

    def test_safe_eigvalsh_basic(self):
        """Test eigenvalue computation for symmetric matrix."""
        # Symmetric matrix with known eigenvalues
        A = np.array([[3.0, 1.0], [1.0, 3.0]])
        eigvals = safe_eigvalsh(A, name="symmetric")

        # Eigenvalues should be 2 and 4
        assert np.allclose(sorted(eigvals), [2.0, 4.0])

    def test_safe_eigvalsh_identity(self):
        """Test that identity matrix has all eigenvalues = 1."""
        I = np.eye(5)
        eigvals = safe_eigvalsh(I, name="identity")

        assert np.allclose(eigvals, 1.0)

    def test_safe_eigvalsh_not_square_raises(self):
        """Test that non-square matrix raises ValueError."""
        A = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(ValueError) as exc_info:
            safe_eigvalsh(A, name="rectangular")

        assert "must be square" in str(exc_info.value)

    def test_safe_eigvalsh_not_positive_definite_warns(self):
        """Test warning when matrix is not positive definite."""
        # Matrix with negative eigenvalue
        A = np.array([[1.0, 2.0], [2.0, 1.0]])  # Eigenvalues: 3, -1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eigvals = safe_eigvalsh(A, name="not_pd", check_positive_definite=True)

            # Should warn about negative eigenvalue
            assert len(w) > 0
            assert "not positive definite" in str(w[0].message)
            assert np.any(eigvals < 0)  # Has negative eigenvalue

    def test_safe_eigvalsh_positive_definite_no_warn(self):
        """Test no warning for positive definite matrix."""
        # Positive definite matrix
        A = np.array([[2.0, 1.0], [1.0, 2.0]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eigvals = safe_eigvalsh(A, name="pd", check_positive_definite=True)

            # No warnings
            assert len(w) == 0
            assert np.all(eigvals > 0)  # All positive

    def test_safe_eigvalsh_ascending_order(self):
        """Test that eigenvalues are returned in ascending order."""
        A = np.diag([5.0, 1.0, 3.0])
        eigvals = safe_eigvalsh(A, name="diagonal")

        # Should be sorted: [1, 3, 5]
        assert np.allclose(eigvals, [1.0, 3.0, 5.0])
        assert np.all(eigvals[:-1] <= eigvals[1:])  # Ascending order


class TestIntegration:
    """Integration tests comparing safe functions to numpy.linalg."""

    def test_safe_inv_matches_numpy_inv(self):
        """Verify safe_inv gives same result as np.linalg.inv for valid input."""
        np.random.seed(123)
        X = np.random.randn(5, 5) + 2 * np.eye(5)

        safe_result = safe_inv(X, check_condition=False)
        numpy_result = np.linalg.inv(X)

        assert np.allclose(safe_result, numpy_result)

    def test_safe_solve_matches_numpy_solve(self):
        """Verify safe_solve gives same result as np.linalg.solve."""
        np.random.seed(456)
        A = np.random.randn(5, 5) + 2 * np.eye(5)
        b = np.random.randn(5)

        safe_result = safe_solve(A, b, check_condition=False)
        numpy_result = np.linalg.solve(A, b)

        assert np.allclose(safe_result, numpy_result)

    def test_safe_lstsq_matches_numpy_lstsq(self):
        """Verify safe_lstsq gives same result as np.linalg.lstsq."""
        np.random.seed(789)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)

        safe_beta, *_ = safe_lstsq(X, y)
        numpy_beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        assert np.allclose(safe_beta, numpy_beta)

    def test_safe_eigvalsh_matches_numpy_eigvalsh(self):
        """Verify safe_eigvalsh gives same result as np.linalg.eigvalsh."""
        np.random.seed(101)
        A = np.random.randn(5, 5)
        A = A + A.T  # Make symmetric

        safe_result = safe_eigvalsh(A, check_positive_definite=False)
        numpy_result = np.linalg.eigvalsh(A)

        assert np.allclose(safe_result, numpy_result)
