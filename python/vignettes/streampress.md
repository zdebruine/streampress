---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# StreamPress I/O

**StreamPress** is a compressed sparse matrix format (`.spz`) designed for large
biological datasets. It achieves **5–10× compression** over raw float32 CSC binary
using rANS entropy coding with delta-encoded row indices, and is faster to read
than raw CSC even single-threaded because sparse object reconstruction is
parallelised across independent chunks.

This vignette uses entirely synthetic data so it runs without any external files.

## Setup

```{code-cell} ipython3
import numpy as np
import scipy.sparse as sp
import streampress as stp
import tempfile, os, time

# Reproducible synthetic sparse matrix — 5000 × 500 with 5% density
# resembles a small single-cell count matrix
rng = np.random.default_rng(42)
m, n = 5000, 500
A = sp.random(m, n, density=0.05, format="csc",
               random_state=rng, dtype=np.float64)
# Simulate count-like values (positive integers)
A.data = np.floor(rng.exponential(5.0, size=A.nnz)) + 1.0
print(f"Matrix: {m:,} rows × {n:,} cols, {A.nnz:,} non-zeros "
      f"({100*A.nnz/(m*n):.1f}% density)")
```

## Writing and Reading

`st_write()` compresses the matrix. `st_read()` decompresses it.

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "counts.spz")

    t0 = time.perf_counter()
    stp.st_write(A, path, include_transpose=True)
    write_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    B = stp.st_read(path)
    read_ms = (time.perf_counter() - t0) * 1000

    info = stp.st_info(path)
    spz_bytes = os.path.getsize(path)
    raw_bytes = A.nnz * 4  # float32 equivalent

    print(f"Write: {write_ms:.0f} ms")
    print(f"Read:  {read_ms:.0f} ms")
    print(f"SPZ file size:      {spz_bytes/1024:.1f} KB")
    print(f"Raw float32 size:   {raw_bytes/1024:.1f} KB")
    print(f"Compression ratio:  {raw_bytes/spz_bytes:.1f}×")
    print(f"Values identical:   {np.allclose(B.toarray(), A.toarray(), rtol=1e-4)}")
```

The `.spz` file is substantially smaller than raw float32 binary. For integer
count data with `precision="fp32"` the compression is lossless because all
integer values ≤ 16,777,216 are representable exactly in float32.

## File Metadata (without decompressing)

`st_info()` reads only the 128-byte header:

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "info.spz")
    stp.st_write(A, path, include_transpose=True)
    info = stp.st_info(path)

for k, v in info.items():
    print(f"  {k:22s} {v}")
```

## Precision Options

StreamPress supports four precision levels:

| Level | Bytes/value | Lossless for integers ≤ |
|-------|-------------|------------------------|
| `fp64` | 8 | 2⁵³ |
| `fp32` | 4 | 2²⁴ = 16,777,216 |
| `fp16` | 2 | 2¹¹ = 2,048 |
| `quant8` | 1 | 256 buckets |

```{code-cell} ipython3
precisions = ["fp64", "fp32", "fp16"]
with tempfile.TemporaryDirectory() as tmp:
    for prec in precisions:
        path = os.path.join(tmp, f"{prec}.spz")
        stp.st_write(A, path, precision=prec, include_transpose=False)
        sz = os.path.getsize(path)
        B = stp.st_read(path)
        err = np.max(np.abs(B.data - A.data))
        print(f"  {prec:6s}  {sz/1024:6.1f} KB   max_err={err:.4f}")
```

The `.spz` format automatically picks `fp32` for typical scRNA-seq count data
when `precision="auto"`, since all UMI counts fit in 24-bit mantissa exactly.

## Column Slicing

Read a column subset without decompressing the entire file:

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "slice.spz")
    stp.st_write(A, path, include_transpose=True)

    # Read columns 100–199
    cols = list(range(100, 200))
    B_cols = stp.st_slice_cols(path, cols)
    print(f"Column slice shape: {B_cols.shape}")
    np.testing.assert_allclose(
        B_cols.toarray(), A[:, 100:200].toarray(), rtol=1e-4)
    print("Values match original ✓")
```

## Row Slicing (via pre-stored transpose)

When written with `include_transpose=True`, row subsets can be read efficiently:

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "rows.spz")
    stp.st_write(A, path, include_transpose=True)

    rows = list(range(0, 500))
    B_rows = stp.st_slice_rows(path, rows)
    print(f"Row slice shape: {B_rows.shape}")
    np.testing.assert_allclose(
        B_rows.toarray(), A[:500, :].toarray(), rtol=1e-4)
    print("Values match original ✓")
```

## Row & Column Metadata

Attach annotation tables (e.g., cell type labels, gene names) to the file:

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "meta.spz")

    # Simulate cell metadata (obs)
    cell_types = ["TypeA" if i % 3 < 2 else "TypeB" for i in range(m)]
    batch = [f"Batch{1+i//1000}" for i in range(m)]
    obs = {"cell_type": cell_types, "batch": batch}

    # Simulate gene metadata (var)
    gene_names = [f"Gene{i}" for i in range(n)]
    var = {"gene_name": gene_names}

    stp.st_write(A, path, obs=obs, var=var, include_transpose=False)
    info = stp.st_info(path)
    print(f"has_obs: {info['has_obs']}  has_var: {info['has_var']}")

    obs_df = stp.st_read_obs(path)
    print(f"obs shape: {obs_df.shape if hasattr(obs_df, 'shape') else len(obs_df)}")
    print(obs_df.head() if hasattr(obs_df, 'head') else list(obs_df.items())[:2])
```

## Streaming Chunk Iteration

`st_map_chunks()` applies a function to each chunk without loading the entire
matrix into memory at once — useful for out-of-core NMF or preprocessing:

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "stream.spz")
    stp.st_write(A, path, include_transpose=False)

    # Count total nnz by streaming
    total_nnz = 0
    n_chunks = 0
    for chunk in stp.st_map_chunks(path, lambda x: x):
        total_nnz += chunk.nnz
        n_chunks += 1

    print(f"Chunks: {n_chunks}")
    print(f"Total nnz via streaming: {total_nnz:,} (expected {A.nnz:,})")
```

## Dense Matrix Support (v3)

StreamPress v3 handles dense matrices with optional compression codecs:

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "dense.spz")

    rng2 = np.random.default_rng(1)
    W = rng2.standard_normal((5000, 20))  # e.g., NMF factorization matrix

    stp.st_write_dense(W, path, codec=0)
    W_back = stp.st_read_dense(path)
    print(f"Dense shape: {W_back.shape}")
    print(f"Max error (fp32 round-trip): {np.max(np.abs(W_back - W)):.6f}")
```

Dense matrices are stored column-major with fp32 precision by default.
The maximum error from the fp64→fp32→fp64 round-trip is ≤ ~6e-8 for
values in the typical NMF range.
