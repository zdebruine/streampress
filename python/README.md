# StreamPress Python Package

High-performance compressed sparse matrix I/O for Python using the
[StreamPress](https://github.com/zdebruine/streampress) `.spz` format.

## Features

- **5–10× compression** over raw float32 CSC binary for sparse matrices
- **Fast reads**: parallelised chunk decompression beats raw CSC even single-threaded
- **Column-oriented chunks**: random column access without full decompression
- **Optional transpose section**: fast row access via pre-stored CSC(Aᵀ)
- **Row/column metadata**: attach `obs`/`var` annotation tables
- **Dense support**: v3 format with FP16/QUANT8/rANS codecs

## Installation

```bash
pip install streampress
```

## Quick Start

```python
import scipy.sparse as sp
import streampress as stp

A = sp.random(10000, 500, density=0.05, format="csc", dtype="float64")

stp.st_write(A, "matrix.spz")
B = stp.st_read("matrix.spz")
info = stp.st_info("matrix.spz")
```

See [https://github.com/zdebruine/streampress](https://github.com/zdebruine/streampress) for full documentation.
