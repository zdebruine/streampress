"""Advanced StreamPress (.spz) operations.

Extends the basic ``st_read``, ``st_write``, ``st_info`` with:
  - Column/row slicing
  - Transpose management
  - Chunk iteration
  - Metadata-based filtering (when obs/var tables are present — future)

Mirrors the R ``streampress.R`` API.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import numpy as np


def _raw_to_csc(raw: dict):
    """Convert raw CSC dict from C++ to scipy.sparse.csc_matrix."""
    from scipy.sparse import csc_matrix

    return csc_matrix(
        (np.asarray(raw["data"]), np.asarray(raw["indices"]), np.asarray(raw["indptr"])),
        shape=(raw["nrow"], raw["ncol"]),
    )


# ---------------------------------------------------------------------------
# Column slicing
# ---------------------------------------------------------------------------


def st_slice_cols(path: str, cols, *, threads: int = 0):
    """Read a subset of columns from a .spz file.

    Parameters
    ----------
    path : str
        Path to the .spz file.
    cols : array-like of int
        Column indices (0-indexed).  When a contiguous range, the C++
        fast-path is used.  Otherwise, the full matrix is read and sliced.
    threads : int
        Decompression threads (0 = all available).

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    from streampress._core import _st_slice_cols, _st_read, _st_info

    cols_arr = np.asarray(cols, dtype=int)

    # Check if contiguous range — can use the C++ chunk-level fast path
    if len(cols_arr) >= 1:
        sorted_cols = np.sort(cols_arr)
        is_contiguous = np.all(np.diff(sorted_cols) == 1)
        if is_contiguous:
            col_start = int(sorted_cols[0])
            col_end = int(sorted_cols[-1]) + 1
            n_requested = col_end - col_start
            raw = _st_slice_cols(path=path, cols=[col_start, col_end], threads=threads)
            mat = _raw_to_csc(raw)
            # C++ decompress_v2 returns entire chunks overlapping the request.
            # If the result is larger, sub-select the exact columns.
            if mat.shape[1] > n_requested:
                # The returned columns start at the first chunk boundary
                # that overlaps our request. Determine that boundary.
                info = dict(_st_info(path=path))
                chunk_cols = int(info.get("chunk_cols", mat.shape[1]))
                first_chunk_start = (col_start // chunk_cols) * chunk_cols
                offset = col_start - first_chunk_start
                mat = mat[:, offset:offset + n_requested]
            return mat

    # Fallback: read full and slice
    raw = _st_read(path=path, threads=threads)
    full = _raw_to_csc(raw)
    return full[:, cols_arr]


# ---------------------------------------------------------------------------
# Row slicing (via pre-stored transpose)
# ---------------------------------------------------------------------------


def st_slice_rows(path: str, rows, *, threads: int = 0):
    """Read a subset of rows from a .spz file.

    Uses the pre-stored transpose section (requires that the file was
    written with ``include_transpose=True``).

    Parameters
    ----------
    path : str
        Path to the .spz file.
    rows : array-like of int
        Row indices (0-indexed).
    threads : int
        Decompression threads.

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    from streampress._core import _st_read_transpose

    rows_arr = np.asarray(rows, dtype=int)
    raw = _st_read_transpose(path=path, threads=threads)
    # Transpose is CSC(A^T) with shape (ncol_orig, nrow_orig)
    At = _raw_to_csc(raw)
    # Columns of A^T = rows of A
    sub_t = At[:, rows_arr]
    return sub_t.T.tocsc()


# ---------------------------------------------------------------------------
# Combined slicing
# ---------------------------------------------------------------------------


def st_slice(
    path: str,
    *,
    rows=None,
    cols=None,
    threads: int = 0,
):
    """Slice rows and/or columns from a .spz file.

    Parameters
    ----------
    path : str
        Path to the .spz file.
    rows : array-like of int or None
        Row indices (0-indexed).
    cols : array-like of int or None
        Column indices (0-indexed).
    threads : int
        Decompression threads.

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    if rows is not None and cols is not None:
        mat = st_slice_cols(path, cols, threads=threads)
        rows_arr = np.asarray(rows, dtype=int)
        return mat[rows_arr, :]
    elif rows is not None:
        return st_slice_rows(path, rows, threads=threads)
    elif cols is not None:
        return st_slice_cols(path, cols, threads=threads)
    else:
        from streampress import st_read
        return st_read(path, threads=threads)


# ---------------------------------------------------------------------------
# Transpose management
# ---------------------------------------------------------------------------


def st_add_transpose(path: str, *, verbose: bool = True) -> bool:
    """Add a pre-computed transpose to an existing .spz file.

    Parameters
    ----------
    path : str
        Path to the .spz file (modified in-place).
    verbose : bool
        Print progress.

    Returns
    -------
    bool
        True if successful.
    """
    from streampress._core import _st_add_transpose

    return _st_add_transpose(path=path, verbose=verbose)


# ---------------------------------------------------------------------------
# Chunk iteration
# ---------------------------------------------------------------------------


def st_chunk_ranges(path: str) -> list:
    """Get column ranges for each chunk in a .spz file.

    Parameters
    ----------
    path : str
        Path to the .spz file.

    Returns
    -------
    list of (int, int)
        List of (start, end) 0-indexed half-open ranges.
    """
    from streampress._core import _st_info

    info = dict(_st_info(path=path))
    chunk_cols = int(info.get("chunk_cols", 0))
    ncol = int(info["ncol"])
    if chunk_cols <= 0:
        return [(0, ncol)]

    ranges = []
    start = 0
    while start < ncol:
        end = min(start + chunk_cols, ncol)
        ranges.append((start, end))
        start = end
    return ranges


def st_map_chunks(
    path: str,
    fn: Callable,
    *,
    transpose: bool = False,
    threads: int = 0,
) -> list:
    """Apply a function to each chunk of a .spz file.

    Parameters
    ----------
    path : str
        Path to the .spz file.
    fn : callable
        Function ``fn(chunk, start, end)`` where *chunk* is a
        ``scipy.sparse.csc_matrix`` and *start*/*end* are column
        (or row) indices.
    transpose : bool
        Iterate over transpose chunks (row chunks).
    threads : int
        Decompression threads.

    Returns
    -------
    list
        Results from each call to *fn*.
    """
    if not transpose:
        ranges = st_chunk_ranges(path)
        results = []
        for start, end in ranges:
            chunk = st_slice_cols(path, list(range(start, end)), threads=threads)
            results.append(fn(chunk, start, end))
        return results
    else:
        from streampress._core import _st_info
        info = dict(_st_info(path=path))
        tc = int(info.get("transp_chunk_cols") or info.get("chunk_cols", 0))
        nrow = int(info["nrow"])
        if tc <= 0:
            tc = nrow

        results = []
        start = 0
        while start < nrow:
            end = min(start + tc, nrow)
            chunk = st_slice_rows(path, list(range(start, end)), threads=threads)
            results.append(fn(chunk, start, end))
            start = end
        return results


# ---------------------------------------------------------------------------
# Metadata-based filtering
# ---------------------------------------------------------------------------


def st_obs_indices(path: str, predicate) -> np.ndarray:
    """Return row indices where *predicate* is True on the obs table.

    Parameters
    ----------
    path : str
        Path to the .spz file.
    predicate : callable
        Function ``predicate(obs_df) -> bool Series`` applied to the obs
        DataFrame. For example: ``lambda df: df["gene_type"] == "protein_coding"``.

    Returns
    -------
    np.ndarray of int
        0-indexed row indices matching the predicate.
    """
    from streampress import st_read_obs

    obs = st_read_obs(path)
    if obs.empty:
        raise ValueError("File has no obs table")
    mask = predicate(obs)
    return np.where(mask)[0]


def st_filter_rows(path: str, predicate, *, threads: int = 0):
    """Read rows matching a predicate on the obs table.

    Parameters
    ----------
    path : str
        Path to the .spz file.
    predicate : callable
        Function ``predicate(obs_df) -> bool Series``.
    threads : int
        Decompression threads.

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    idx = st_obs_indices(path, predicate)
    if len(idx) == 0:
        raise ValueError("No rows match filter criteria")
    return st_slice_rows(path, idx.tolist(), threads=threads)


def st_filter_cols(path: str, predicate, *, threads: int = 0):
    """Read columns matching a predicate on the var table.

    Parameters
    ----------
    path : str
        Path to the .spz file.
    predicate : callable
        Function ``predicate(var_df) -> bool Series``.
    threads : int
        Decompression threads.

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    from streampress import st_read_var

    var = st_read_var(path)
    if var.empty:
        raise ValueError("File has no var table")
    mask = predicate(var)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError("No columns match filter criteria")
    return st_slice_cols(path, idx.tolist(), threads=threads)


def st_write_list(
    matrices,
    path: str,
    *,
    chunk_cols: int = 0,
    verbose: bool = False,
) -> None:
    """Column-bind a list of sparse matrices and write to .spz format.

    Parameters
    ----------
    matrices : list of scipy.sparse.spmatrix
        Sparse matrices to concatenate. All must have the same number of rows.
    path : str
        Output file path.
    chunk_cols : int
        Columns per chunk (0 = auto).
    verbose : bool
        Print progress.
    """
    from scipy.sparse import hstack
    from streampress import st_write

    if not matrices:
        raise ValueError("matrices list must not be empty")
    combined = hstack(matrices, format="csc")
    st_write(combined, path, chunk_cols=chunk_cols, verbose=verbose)
