"""
pytest tests for the streampress Python package.

All tests use synthetic data so they run anywhere with no external files.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import tempfile
import os

import streampress as stp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sparse(m=500, n=100, density=0.05, seed=42):
    rng = np.random.default_rng(seed)
    return sp.random(m, n, density=density, format="csc",
                     dtype=np.float64, random_state=rng)


def make_count_matrix(m=200, n=50, seed=0):
    """Integer count matrix (like scRNA-seq UMI counts)."""
    rng = np.random.default_rng(seed)
    A = sp.random(m, n, density=0.1, format="csc",
                  random_state=rng)
    A.data = np.floor(rng.exponential(5.0, size=A.nnz)) + 1.0
    return A


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_basic_roundtrip(self, tmp_path):
        A = make_sparse()
        path = str(tmp_path / "test.spz")
        stp.st_write(A, path, include_transpose=False)
        B = stp.st_read(path)
        assert B.shape == A.shape
        assert B.nnz == A.nnz
        np.testing.assert_allclose(B.toarray(), A.toarray(), rtol=1e-5)

    def test_roundtrip_with_transpose(self, tmp_path):
        A = make_sparse(m=200, n=80)
        path = str(tmp_path / "test_t.spz")
        stp.st_write(A, path, include_transpose=True)
        B = stp.st_read(path)
        assert B.shape == A.shape
        np.testing.assert_allclose(B.toarray(), A.toarray(), rtol=1e-5)

    def test_count_matrix(self, tmp_path):
        A = make_count_matrix()
        path = str(tmp_path / "counts.spz")
        stp.st_write(A, path, precision="fp32", include_transpose=False)
        B = stp.st_read(path)
        assert B.shape == A.shape
        # fp32 is lossless for integers up to 2^24
        np.testing.assert_allclose(B.toarray(), A.toarray(), rtol=1e-4)

    def test_single_column(self, tmp_path):
        A = make_sparse(m=100, n=1)
        path = str(tmp_path / "single.spz")
        stp.st_write(A, path, include_transpose=False)
        B = stp.st_read(path)
        assert B.shape == A.shape

    def test_empty_matrix(self, tmp_path):
        A = sp.csc_matrix((50, 20))
        path = str(tmp_path / "empty.spz")
        stp.st_write(A, path, include_transpose=False)
        B = stp.st_read(path)
        assert B.shape == A.shape
        assert B.nnz == 0

    @pytest.mark.parametrize("precision", ["auto", "fp32", "fp64"])
    def test_precision_roundtrip(self, tmp_path, precision):
        A = make_sparse(m=100, n=30)
        path = str(tmp_path / f"prec_{precision}.spz")
        stp.st_write(A, path, precision=precision, include_transpose=False)
        B = stp.st_read(path)
        np.testing.assert_allclose(B.toarray(), A.toarray(), rtol=1e-4)


# ---------------------------------------------------------------------------
# Info tests
# ---------------------------------------------------------------------------

class TestInfo:

    def test_info_dimensions(self, tmp_path):
        A = make_sparse(m=300, n=50)
        path = str(tmp_path / "info.spz")
        stp.st_write(A, path, include_transpose=False)
        info = stp.st_info(path)
        assert info["nrow"] == 300
        assert info["ncol"] == 50
        assert info["nnz"] == A.nnz
        assert info["version"] >= 2
        assert not info["has_transpose"]

    def test_info_with_transpose(self, tmp_path):
        A = make_sparse(m=100, n=40)
        path = str(tmp_path / "info_t.spz")
        stp.st_write(A, path, include_transpose=True)
        info = stp.st_info(path)
        assert info["has_transpose"]


# ---------------------------------------------------------------------------
# Column slicing
# ---------------------------------------------------------------------------

class TestSlice:

    def test_slice_cols_contiguous(self, tmp_path):
        A = make_sparse(m=200, n=100)
        path = str(tmp_path / "slice.spz")
        stp.st_write(A, path, include_transpose=False)
        B = stp.st_slice_cols(path, list(range(10, 30)))
        assert B.shape == (200, 20)
        np.testing.assert_allclose(
            B.toarray(), A[:, 10:30].toarray(), rtol=1e-4)

    def test_slice_rows(self, tmp_path):
        A = make_sparse(m=200, n=50)
        path = str(tmp_path / "slice_r.spz")
        stp.st_write(A, path, include_transpose=True)
        B = stp.st_slice_rows(path, list(range(0, 50)))
        assert B.shape[0] == 50
        assert B.shape[1] == 50

    def test_slice_combined(self, tmp_path):
        A = make_sparse(m=200, n=100)
        path = str(tmp_path / "slice_comb.spz")
        stp.st_write(A, path, include_transpose=True)
        B = stp.st_slice(path, cols=list(range(20, 40)))
        assert B.shape[1] == 20


# ---------------------------------------------------------------------------
# Dense round-trip
# ---------------------------------------------------------------------------

class TestDense:

    def test_dense_roundtrip(self, tmp_path):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 50))
        path = str(tmp_path / "dense.spz")
        stp.st_write_dense(X, path, codec=0)
        Y = stp.st_read_dense(path)
        np.testing.assert_allclose(Y, X, rtol=1e-5)

    def test_dense_shape(self, tmp_path):
        X = np.ones((30, 20))
        path = str(tmp_path / "dense_ones.spz")
        stp.st_write_dense(X, path)
        Y = stp.st_read_dense(path)
        assert Y.shape == X.shape


# ---------------------------------------------------------------------------
# Metadata tables
# ---------------------------------------------------------------------------

class TestMetadata:

    def test_write_read_obs(self, tmp_path):
        A = make_sparse(m=100, n=50)
        obs = {
            "cell_type": ["A" if i % 2 == 0 else "B" for i in range(100)],
        }
        path = str(tmp_path / "meta.spz")
        stp.st_write(A, path, obs=obs, include_transpose=False)
        info = stp.st_info(path)
        assert info["has_obs"]
        obs_back = stp.st_read_obs(path)
        assert "cell_type" in obs_back
        assert len(obs_back["cell_type"]) == 100

    def test_no_metadata(self, tmp_path):
        A = make_sparse()
        path = str(tmp_path / "nometa.spz")
        stp.st_write(A, path, include_transpose=False)
        info = stp.st_info(path)
        assert not info["has_obs"]
        assert not info["has_var"]


# ---------------------------------------------------------------------------
# Compression check
# ---------------------------------------------------------------------------

class TestCompression:

    def test_file_smaller_than_raw(self, tmp_path):
        """SPZ file should be smaller than raw float32 CSC binary."""
        A = make_count_matrix(m=2000, n=200)
        path = str(tmp_path / "compress.spz")
        stp.st_write(A, path, include_transpose=False)
        spz_bytes = os.path.getsize(path)
        raw_bytes = A.nnz * 4  # float32
        assert spz_bytes < raw_bytes, (
            f"SPZ ({spz_bytes}) should be smaller than raw float32 ({raw_bytes})")
