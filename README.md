# StreamPress

[![Python tests](https://github.com/zdebruine/streampress/actions/workflows/python-tests.yaml/badge.svg)](https://github.com/zdebruine/streampress/actions/workflows/python-tests.yaml) [![R-CMD-check](https://github.com/zdebruine/streampress/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/zdebruine/streampress/actions/workflows/R-CMD-check.yaml)

**StreamPress** is a header-only C++ library (with Python and R bindings) for high-performance compressed sparse matrix I/O using the `.spz` file format.

**📖 Documentation:** [**Python**](https://zdebruine.github.io/streampress/python/) · [**R**](https://zdebruine.github.io/streampress/r/) · [**C++ headers**](include/streampress/)

## Why StreamPress?

Standard sparse matrix serialization formats (`.rds`, `.pickle`, Matlab `.mat`) are general-purpose and inefficient for large biological datasets. StreamPress is designed specifically for large sparse floating-point matrices such as single-cell RNA-seq count data:

- **5–10× compression** over raw float32 CSC binary using rANS entropy coding with delta-encoded indices
- **Faster reads** than raw CSC binary even single-threaded — SPZ parallelises sparse object reconstruction across independent chunks
- **Column-oriented chunking** enables random column access and out-of-core streaming without decompressing the full file
- **Optional transpose storage** — pre-store CSC(Aᵀ) for fast row access
- **Row/column metadata** — attach `obs` (cell) and `var` (gene) annotation tables to the file
- **Dense matrix support** — optional FP16, QUANT8, and rANS codecs for dense data

## Installation

### C++ (header-only)

Copy `include/streampress/` into your project, or use as a Meson subproject:

```ini
# subprojects/streampress.wrap
[wrap-git]
url = https://github.com/zdebruine/streampress.git
revision = main
depth = 1
```

```meson
streampress_dep = dependency('streampress',
  fallback: ['streampress', 'streampress_dep'])
```

No external dependencies — StreamPress only requires C++17 and standard headers.

### Python

```bash
pip install streampress
```

Requires Python ≥ 3.9, numpy, scipy.

### R

```r
install.packages("streampress")
# Development version:
remotes::install_github("zdebruine/streampress", subdir = "r")
```

## Quick Start

### C++

```cpp
#include <streampress/streampress_api.hpp>

// Build a sparse matrix (CSC format)
streampress::CSCMatrix mat;
mat.m = 4; mat.n = 3; mat.nnz = 6;
mat.p = {0, 2, 3, 6};
mat.i = {0, 2, 1, 0, 2, 3};
mat.x = {1.0, 4.0, 3.0, 2.0, 5.0, 6.0};

// Write to file
streampress::api::WriteOptions opts;
streampress::api::write_sparse("matrix.spz", mat, {}, {}, opts);

// Read back
auto result = streampress::api::read_sparse("matrix.spz");
// result.m, result.n, result.nnz, result.col_ptr, result.row_ind, result.values

// Inspect metadata (no decompression)
auto info = streampress::api::info("matrix.spz");
// info.m, info.n, info.nnz, info.chunk_cols, info.has_transpose, ...

// Read a column range
auto cols = streampress::api::slice_cols("matrix.spz", 0, 2);
```

### Python

```python
import scipy.sparse as sp
import streampress as stp

# Write a sparse matrix
A = sp.random(10000, 500, density=0.05, format="csc", dtype="float64")
stp.st_write(A, "matrix.spz")

# Read back
B = stp.st_read("matrix.spz")

# Inspect metadata (fast, no decompression)
info = stp.st_info("matrix.spz")
print(f"{info['nrow']}×{info['ncol']}, ratio={info['density']:.4f}")

# Read a column subset
cols = stp.st_slice_cols("matrix.spz", list(range(0, 100)))

# Streaming chunk iteration
for chunk in stp.st_map_chunks("matrix.spz", lambda x: x):
    process(chunk)
```

### R

```r
library(streampress)
library(Matrix)

A <- rsparsematrix(10000, 500, density = 0.05)

# Write
st_write(A, "matrix.spz")

# Read
B <- st_read("matrix.spz")

# Inspect (fast)
info <- st_info("matrix.spz")

# Partial read
B_sub <- st_read("matrix.spz", cols = 1:100)
```

## Compression Pipeline

For each column chunk, StreamPress applies:

1. **Precision reduction** — optionally convert fp64 → fp32/fp16/quant8 before encoding
2. **Delta encoding** — store row-index differences rather than absolute indices (sparse count data usually has deltas of 1–5, compressing extremely well)
3. **rANS entropy coding** — asymmetric numeral system approaching the Shannon entropy bound

## Chunk Architecture

```
┌─────────────────────────────────────────────┐
│  File Header (128 bytes)                    │
│  – dimensions, nnz, format version          │
│  – chunk index (offset + size per chunk)    │
│  – obs/var table offsets                    │
├─────────────────────────────────────────────┤
│  Chunk 0  │  Chunk 1  │  ...  │  Chunk N-1  │
│  (nnz₀ values + indices, rANS-compressed)   │
├─────────────────────────────────────────────┤
│  Transpose Section (optional)               │
│  – CSC(Aᵀ) stored independently            │
├─────────────────────────────────────────────┤
│  obs table (optional)  │  var table (opt)   │
└─────────────────────────────────────────────┘
```

## API Reference

### C++ (`namespace streampress::api`)

| Function | Description |
|----------|-------------|
| `write_sparse(path, mat, obs, var, opts)` | Compress CSC matrix to `.spz` |
| `read_sparse(path, opts)` | Decompress `.spz` to `ReadResult` |
| `slice_cols(path, start, end, opts)` | Read column range [start, end) |
| `info(path)` | Read file metadata (no decompression) |
| `add_transpose(path)` | Add pre-computed transpose section |

### Python (`streampress`)

| Function | Description |
|----------|-------------|
| `st_write(mat, path, **kwargs)` | Write `scipy.sparse.csc_matrix` |
| `st_read(path, **kwargs)` | Read `.spz` → `csc_matrix` |
| `st_info(path)` | Read metadata dict |
| `st_slice_cols(path, cols)` | Read column subset |
| `st_slice_rows(path, rows)` | Read row subset (needs transpose) |
| `st_slice(path, rows, cols)` | Read submatrix |
| `st_add_transpose(path)` | Add transpose section |
| `st_map_chunks(path, fn)` | Apply function to each chunk |
| `st_write_dense(X, path)` | Write dense numpy array |
| `st_read_dense(path)` | Read dense file |

### R (`streampress`)

| Function | Description |
|----------|-------------|
| `st_write(x, path, ...)` | Write `dgCMatrix` to `.spz` |
| `st_read(path, ...)` | Read `.spz` → `dgCMatrix` |
| `st_info(path)` | Read file metadata list |
| `st_write_dense(x, path)` | Write dense matrix |
| `st_read_dense(path)` | Read dense file |
| `st_read_obs(path)` | Read row metadata `data.frame` |
| `st_read_var(path)` | Read column metadata `data.frame` |
| `st_convert(input, output, ...)` | Convert to `.spz` with options |

## Repository Structure

```
streampress/
├── include/streampress/    ← Header-only C++ library
│   ├── streampress_api.hpp ← Public API (write, read, slice, info)
│   ├── sparse.hpp          ← Sparse chunked format (primary)
│   ├── dense.hpp           ← Dense column-panel format
│   ├── codec/              ← rANS, Golomb-Rice, VarInt, bitstream
│   ├── core/               ← CSCMatrix, PRNG, platform I/O
│   ├── format/             ← Binary header structures
│   ├── model/              ← Compressor models
│   └── transform/          ← Delta encoding, value mapping
├── tests/                  ← C++ unit tests (meson)
├── python/                 ← PyPI package (scikit-build-core + nanobind)
│   └── src/streampress/
└── r/                      ← CRAN package (Rcpp)
    ├── R/
    └── src/
```

## License

MIT — see [LICENSE](LICENSE).
