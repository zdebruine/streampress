/**
 * @file test_streampress.cpp
 * @brief Unit tests for StreamPress round-trip encode/decode.
 */

#include <streampress/streampress_api.hpp>
#include <streampress/sparsepress_v3.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

static streampress::CSCMatrix make_4x3() {
    // [1 0 2]
    // [0 3 0]
    // [4 0 5]
    // [0 0 6]
    streampress::CSCMatrix m;
    m.m   = 4;
    m.n   = 3;
    m.nnz = 6;
    m.p   = {0, 2, 3, 6};
    m.i   = {0, 2, 1, 0, 2, 3};
    m.x   = {1.0, 4.0, 3.0, 2.0, 5.0, 6.0};
    return m;
}

// ---------------------------------------------------------------------------
// Test 1: basic sparse round-trip
// ---------------------------------------------------------------------------

static void test_sparse_roundtrip() {
    const char* path = "/tmp/sp_test_roundtrip.spz";
    auto mat = make_4x3();

    streampress::api::WriteOptions opts;
    opts.chunk_cols = 2;
    opts.include_transpose = false;
    streampress::api::write_sparse(path, mat, {}, {}, opts);

    streampress::api::ReadOptions ropts;
    ropts.threads = 1;
    auto r = streampress::api::read_sparse(path, ropts);

    check(r.m == mat.m, "roundtrip: rows");
    check(r.n == mat.n, "roundtrip: cols");
    check(r.nnz == mat.nnz, "roundtrip: nnz");
    for (size_t k = 0; k < mat.x.size(); ++k) {
        float diff = r.values[k] - static_cast<float>(mat.x[k]);
        check(std::fabs(diff) < 1e-4f, "roundtrip: values");
    }
    std::remove(path);
    std::cout << "PASS: sparse_roundtrip\n";
}

// ---------------------------------------------------------------------------
// Test 2: column slicing
// ---------------------------------------------------------------------------

static void test_slice_cols() {
    const char* path = "/tmp/sp_test_slice.spz";
    auto mat = make_4x3();

    streampress::api::WriteOptions opts;
    opts.chunk_cols = 1;
    opts.include_transpose = false;
    streampress::api::write_sparse(path, mat, {}, {}, opts);

    // Read columns 0–1 (first 2 columns only)
    auto r = streampress::api::slice_cols(path, 0, 2);
    check(r.n == 2, "slice_cols: ncol");
    check(r.m == 4, "slice_cols: nrow");
    // col 0 has values {1.0, 4.0} at rows {0, 2}
    check(r.nnz == 3, "slice_cols: nnz (2 from col0 + 1 from col1)");

    std::remove(path);
    std::cout << "PASS: slice_cols\n";
}

// ---------------------------------------------------------------------------
// Test 3: file info (no decompression)
// ---------------------------------------------------------------------------

static void test_info() {
    const char* path = "/tmp/sp_test_info.spz";
    auto mat = make_4x3();

    streampress::api::WriteOptions opts;
    opts.chunk_cols = 2;
    opts.include_transpose = false;
    streampress::api::write_sparse(path, mat, {}, {}, opts);

    auto fi = streampress::api::info(path);
    check(fi.m == 4, "info: rows");
    check(fi.n == 3, "info: cols");
    check(fi.nnz == 6, "info: nnz");
    check(fi.version == 2, "info: version");
    check(!fi.has_transpose, "info: no transpose");

    std::remove(path);
    std::cout << "PASS: info\n";
}

// ---------------------------------------------------------------------------
// Test 4: with transpose
// ---------------------------------------------------------------------------

static void test_with_transpose() {
    const char* path = "/tmp/sp_test_transp.spz";
    auto mat = make_4x3();

    streampress::api::WriteOptions opts;
    opts.chunk_cols = 2;
    opts.include_transpose = true;
    streampress::api::write_sparse(path, mat, {}, {}, opts);

    auto fi = streampress::api::info(path);
    check(fi.has_transpose, "transpose: flag set");

    std::remove(path);
    std::cout << "PASS: with_transpose\n";
}

// ---------------------------------------------------------------------------
// Test 5: precision — fp32 round-trip
// ---------------------------------------------------------------------------

static void test_precision_fp32() {
    const char* path = "/tmp/sp_test_fp32.spz";
    auto mat = make_4x3();

    streampress::api::WriteOptions opts;
    opts.precision = "fp32";
    opts.chunk_cols = 2;
    opts.include_transpose = false;
    streampress::api::write_sparse(path, mat, {}, {}, opts);

    auto r = streampress::api::read_sparse(path);
    check(r.nnz == mat.nnz, "fp32: nnz");
    for (size_t k = 0; k < mat.x.size(); ++k) {
        float diff = r.values[k] - static_cast<float>(mat.x[k]);
        check(std::fabs(diff) < 1e-4f, "fp32: values");
    }
    std::remove(path);
    std::cout << "PASS: precision_fp32\n";
}

// ---------------------------------------------------------------------------
// Test 6: larger random-ish matrix
// ---------------------------------------------------------------------------

static void test_larger_matrix() {
    const char* path = "/tmp/sp_test_large.spz";

    streampress::CSCMatrix mat;
    mat.m = 1000;
    mat.n = 20;
    mat.p.resize(mat.n + 1, 0);

    // ~10% density, values 1–100
    for (uint32_t j = 0; j < mat.n; ++j) {
        uint32_t start = static_cast<uint32_t>(mat.i.size());
        for (uint32_t i = 0; i < mat.m; ++i) {
            // deterministic "random" via xorshift
            uint64_t h = (static_cast<uint64_t>(i) * 6364136223846793005ULL +
                          static_cast<uint64_t>(j) * 1442695040888963407ULL);
            if ((h >> 32) % 10 == 0) {
                mat.i.push_back(i);
                mat.x.push_back(static_cast<double>((h % 100) + 1));
            }
        }
        mat.p[j + 1] = static_cast<uint32_t>(mat.i.size());
    }
    mat.nnz = mat.i.size();

    streampress::api::WriteOptions opts;
    opts.chunk_cols = 5;
    opts.include_transpose = false;
    streampress::api::write_sparse(path, mat, {}, {}, opts);

    auto r = streampress::api::read_sparse(path);
    check(r.m == mat.m, "larger: rows");
    check(r.n == mat.n, "larger: cols");
    check(r.nnz == mat.nnz, "larger: nnz");
    for (size_t k = 0; k < mat.x.size(); ++k) {
        float diff = r.values[k] - static_cast<float>(mat.x[k]);
        check(std::fabs(diff) < 0.5f, "larger: values");
    }

    std::remove(path);
    std::cout << "PASS: larger_matrix (m=" << mat.m << " n=" << mat.n
              << " nnz=" << mat.nnz << ")\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== StreamPress C++ tests ===\n";
    test_sparse_roundtrip();
    test_slice_cols();
    test_info();
    test_with_transpose();
    test_precision_fp32();
    test_larger_matrix();
    std::cout << "=== All tests passed ===\n";
    return 0;
}
