/**
 * @file _bindings.cpp
 * @brief nanobind bindings for StreamPress (.spz) I/O operations.
 *
 * Exposes the StreamPress C++ API to Python as streampress._core.
 * All public functions are prefixed with _st_ and wrapped by the
 * streampress Python module (streampress/__init__.py).
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <streampress/streampress_api.hpp>
#include <streampress/transpose.hpp>
#include <streampress/dense.hpp>
#include <streampress/format/header_sparse.hpp>
#include <streampress/format/obs_var_table.hpp>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace nb = nanobind;
using namespace nb::literals;

// =============================================================================
// Read sparse
// =============================================================================

static nb::dict py_st_read(const std::string& path, int threads, bool reorder) {
    streampress::api::ReadOptions opts;
    opts.threads = threads;
    opts.reorder = reorder;

    auto rr = streampress::api::read_sparse(path, opts);

    nb::dict ret;
    ret["nrow"] = rr.m;
    ret["ncol"] = rr.n;
    ret["nnz"]  = static_cast<uint64_t>(rr.nnz);

    const size_t nnz  = rr.nnz;
    const size_t ncol = rr.n;

    // indptr (uint32 → int32 for scipy compatibility)
    {
        auto* buf = new int32_t[ncol + 1];
        for (size_t j = 0; j <= ncol; ++j)
            buf[j] = static_cast<int32_t>(rr.col_ptr[j]);
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
        size_t shape[1] = {ncol + 1};
        ret["indptr"] = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(buf, 1, shape, owner);
    }
    // indices (uint32 → int32)
    {
        auto* buf = new int32_t[nnz];
        for (size_t k = 0; k < nnz; ++k)
            buf[k] = static_cast<int32_t>(rr.row_ind[k]);
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
        size_t shape[1] = {nnz};
        ret["indices"] = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(buf, 1, shape, owner);
    }
    // data (float32 → float64 for scipy)
    {
        auto* buf = new double[nnz];
        for (size_t k = 0; k < nnz; ++k)
            buf[k] = static_cast<double>(rr.values[k]);
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        size_t shape[1] = {nnz};
        ret["data"] = nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
    }

    return ret;
}

// =============================================================================
// Write sparse
// =============================================================================

static void py_st_write(
    int nrow, int ncol,
    nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig> indptr,
    nb::ndarray<int32_t, nb::ndim<1>, nb::c_contig> indices,
    nb::ndarray<double,  nb::ndim<1>, nb::c_contig> data,
    const std::string& path,
    int chunk_cols,
    bool verbose,
    bool include_transpose,
    bool use_delta,
    const std::string& precision,
    int threads,
    nb::object obs_dict,
    nb::object var_dict,
    bool value_pred,
    bool row_sort,
    int64_t chunk_bytes,
    int transp_chunk_cols)
{
    const size_t nnz = data.shape(0);

    streampress::CSCMatrix mat;
    mat.m   = static_cast<uint32_t>(nrow);
    mat.n   = static_cast<uint32_t>(ncol);
    mat.nnz = static_cast<uint64_t>(nnz);

    mat.p.resize(ncol + 1);
    for (int j = 0; j <= ncol; ++j)
        mat.p[j] = static_cast<uint32_t>(indptr.data()[j]);

    mat.i.resize(nnz);
    for (size_t k = 0; k < nnz; ++k)
        mat.i[k] = static_cast<uint32_t>(indices.data()[k]);

    mat.x.resize(nnz);
    for (size_t k = 0; k < nnz; ++k)
        mat.x[k] = data.data()[k];

    streampress::api::WriteOptions opts;
    if (chunk_cols > 0) opts.chunk_cols = static_cast<uint32_t>(chunk_cols);
    opts.include_transpose = include_transpose;
    opts.use_delta = use_delta;
    opts.precision = precision;
    opts.threads = threads;
    opts.value_pred = value_pred;
    opts.row_sort = row_sort;
    if (chunk_bytes > 0) opts.chunk_bytes = static_cast<uint64_t>(chunk_bytes);
    if (transp_chunk_cols > 0) opts.transp_chunk_cols = static_cast<uint32_t>(transp_chunk_cols);

    // Helper: convert Python dict-of-arrays → serialized table buffer
    auto dict_to_table_buf = [](nb::object dict_obj, uint32_t n_rows)
        -> std::vector<uint8_t>
    {
        if (dict_obj.is_none() || nb::len(dict_obj) == 0) return {};
        nb::dict d = nb::cast<nb::dict>(dict_obj);
        std::vector<streampress::v2::ColumnData> columns;
        for (auto [key, val] : d) {
            std::string col_name = nb::cast<std::string>(key);
            streampress::v2::ColumnData cd;
            cd.name = col_name;
            try {
                auto arr = nb::cast<nb::ndarray<double, nb::ndim<1>>>(val);
                cd.type = streampress::v2::ColType::FLOAT64;
                cd.dbl_data.resize(n_rows);
                for (uint32_t i = 0; i < n_rows; ++i)
                    cd.dbl_data[i] = arr.data()[i];
            } catch (...) {
                try {
                    auto arr = nb::cast<nb::ndarray<int32_t, nb::ndim<1>>>(val);
                    cd.type = streampress::v2::ColType::INT32;
                    cd.int_data.resize(n_rows);
                    for (uint32_t i = 0; i < n_rows; ++i)
                        cd.int_data[i] = arr.data()[i];
                } catch (...) {
                    cd.type = streampress::v2::ColType::STRING_DICT;
                    nb::list lst = nb::cast<nb::list>(val);
                    std::unordered_map<std::string, uint32_t> str_map;
                    cd.codes.resize(n_rows);
                    for (uint32_t i = 0; i < n_rows; ++i) {
                        std::string s = nb::cast<std::string>(lst[i]);
                        auto it = str_map.find(s);
                        if (it == str_map.end()) {
                            uint32_t idx = static_cast<uint32_t>(cd.dict.size());
                            str_map[s] = idx;
                            cd.dict.push_back(s);
                            cd.codes[i] = idx;
                        } else {
                            cd.codes[i] = it->second;
                        }
                    }
                }
            }
            columns.push_back(std::move(cd));
        }
        return streampress::v2::obs_var_table_serialize(n_rows, columns);
    };

    std::vector<uint8_t> obs_buf = dict_to_table_buf(obs_dict, static_cast<uint32_t>(nrow));
    std::vector<uint8_t> var_buf = dict_to_table_buf(var_dict, static_cast<uint32_t>(ncol));

    streampress::api::write_sparse(path, mat, obs_buf, var_buf, opts);
}

// =============================================================================
// Info (no decompression)
// =============================================================================

static nb::dict py_st_info(const std::string& path) {
    auto fi = streampress::api::info(path);
    nb::dict ret;
    ret["nrow"]            = fi.m;
    ret["ncol"]            = fi.n;
    ret["nnz"]             = static_cast<uint64_t>(fi.nnz);
    ret["version"]         = static_cast<int>(fi.version);
    ret["density"]         = static_cast<double>(fi.density);
    ret["file_bytes"]      = static_cast<uint64_t>(fi.file_bytes);
    ret["has_transpose"]   = fi.has_transpose;
    ret["chunk_cols"]      = fi.chunk_cols;
    ret["transp_chunk_cols"] = fi.transp_chunk_cols;
    ret["has_obs"]         = fi.has_obs_table;
    ret["has_var"]         = fi.has_var_table;
    return ret;
}

// =============================================================================
// Read transpose
// =============================================================================

static nb::dict py_st_read_transpose(const std::string& path, int threads) {
    std::vector<uint8_t> data = streampress::v2::read_v2(path);
    streampress::v2::DecompressConfig_v2 dcfg;
    dcfg.num_threads = threads;
    streampress::CSCMatrix mat =
        streampress::v2::decompress_v2_transpose(data.data(), data.size(), dcfg);

    nb::dict ret;
    ret["nrow"] = mat.m;
    ret["ncol"] = mat.n;
    uint64_t nnz = mat.i.size();
    ret["nnz"]  = nnz;

    {
        size_t ncol = mat.n;
        auto* buf = new int32_t[ncol + 1];
        for (size_t j = 0; j <= ncol; ++j)
            buf[j] = static_cast<int32_t>(mat.p[j]);
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
        size_t shape[1] = {ncol + 1};
        ret["indptr"] = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(buf, 1, shape, owner);
    }
    {
        auto* buf = new int32_t[nnz];
        for (size_t k = 0; k < nnz; ++k)
            buf[k] = static_cast<int32_t>(mat.i[k]);
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
        size_t shape[1] = {static_cast<size_t>(nnz)};
        ret["indices"] = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(buf, 1, shape, owner);
    }
    {
        auto* buf = new double[nnz];
        for (size_t k = 0; k < nnz; ++k)
            buf[k] = static_cast<double>(mat.x[k]);
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        size_t shape[1] = {static_cast<size_t>(nnz)};
        ret["data"] = nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
    }
    return ret;
}

// =============================================================================
// Slice columns
// =============================================================================

static nb::dict py_st_slice_cols(const std::string& path,
                                  std::vector<int> cols, int threads) {
    if (cols.size() != 2)
        throw std::invalid_argument("cols must be [start, end) range (0-indexed)");

    uint32_t col_start = static_cast<uint32_t>(cols[0]);
    uint32_t col_end   = static_cast<uint32_t>(cols[1]);

    streampress::api::ReadOptions opts;
    opts.threads = threads;
    auto rr = streampress::api::slice_cols(path, col_start, col_end, opts);

    nb::dict ret;
    ret["nrow"] = rr.m;
    ret["ncol"] = rr.n;
    ret["nnz"]  = static_cast<uint64_t>(rr.nnz);

    const size_t nnz  = rr.nnz;
    const size_t ncol = rr.n;

    {
        auto* buf = new int32_t[ncol + 1];
        for (size_t j = 0; j <= ncol; ++j)
            buf[j] = static_cast<int32_t>(rr.col_ptr[j]);
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
        size_t shape[1] = {ncol + 1};
        ret["indptr"] = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(buf, 1, shape, owner);
    }
    {
        auto* buf = new int32_t[nnz];
        for (size_t k = 0; k < nnz; ++k)
            buf[k] = static_cast<int32_t>(rr.row_ind[k]);
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
        size_t shape[1] = {nnz};
        ret["indices"] = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(buf, 1, shape, owner);
    }
    {
        auto* buf = new double[nnz];
        for (size_t k = 0; k < nnz; ++k)
            buf[k] = static_cast<double>(rr.values[k]);
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        size_t shape[1] = {nnz};
        ret["data"] = nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
    }
    return ret;
}

// =============================================================================
// Add transpose
// =============================================================================

static bool py_st_add_transpose(const std::string& path, bool verbose) {
    return streampress::add_transpose(path, verbose);
}

// =============================================================================
// Dense write
// =============================================================================

static nb::dict py_st_write_dense(
    nb::ndarray<double, nb::ndim<2>, nb::c_contig> X,
    const std::string& path,
    bool include_transpose, int chunk_cols, int codec, bool delta)
{
    const uint32_t m = static_cast<uint32_t>(X.shape(0));
    const uint32_t n = static_cast<uint32_t>(X.shape(1));

    // Convert row-major double → column-major float
    std::vector<float> fdata(static_cast<size_t>(m) * n);
    const double* src = X.data();
    for (uint32_t j = 0; j < n; ++j)
        for (uint32_t i = 0; i < m; ++i)
            fdata[static_cast<size_t>(j) * m + i] =
                static_cast<float>(src[static_cast<size_t>(i) * n + j]);

    streampress::v3::DenseCodec dc =
        static_cast<streampress::v3::DenseCodec>(codec);

    streampress::v3::write_v3<float>(
        path, fdata.data(), m, n,
        static_cast<uint32_t>(chunk_cols), include_transpose, dc, delta);

    nb::dict ret;
    ret["rows"]    = static_cast<int>(m);
    ret["cols"]    = static_cast<int>(n);
    ret["version"] = 3;
    return ret;
}

// =============================================================================
// Dense read
// =============================================================================

static nb::ndarray<nb::numpy, double, nb::ndim<2>> py_st_read_dense(
    const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    std::fseek(f, 0, SEEK_END);
    size_t file_size = static_cast<size_t>(std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(file_size);
    if (std::fread(buf.data(), 1, file_size, f) != file_size) {
        std::fclose(f);
        throw std::runtime_error("Failed to read file: " + path);
    }
    std::fclose(f);

    uint16_t ver = streampress::v3::detect_version(buf.data(), file_size);
    if (ver != 3)
        throw std::runtime_error("Not a dense .spz file (version=" +
                                 std::to_string(ver) + ")");

    std::vector<float> fdata;
    uint32_t m, n;
    streampress::v3::read_full_matrix<float>(buf.data(), file_size, fdata, m, n);

    // Convert column-major float → row-major double for numpy
    auto* out = new double[static_cast<size_t>(m) * n];
    for (uint32_t j = 0; j < n; ++j)
        for (uint32_t i = 0; i < m; ++i)
            out[static_cast<size_t>(i) * n + j] =
                static_cast<double>(fdata[static_cast<size_t>(j) * m + i]);

    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    size_t shape[2] = {m, n};
    return nb::ndarray<nb::numpy, double, nb::ndim<2>>(out, 2, shape, owner);
}

// =============================================================================
// Read obs/var tables
// =============================================================================

static nb::dict py_st_read_table(const std::string& path, bool is_obs) {
    using namespace streampress::v2;

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    uint8_t hdr_buf[HEADER_SIZE_V2];
    if (std::fread(hdr_buf, 1, HEADER_SIZE_V2, f) != HEADER_SIZE_V2) {
        std::fclose(f);
        throw std::runtime_error("Failed to read header");
    }

    FileHeader_v2 hdr = FileHeader_v2::deserialize(hdr_buf);
    if (!hdr.valid() || hdr.version != 2) {
        std::fclose(f);
        throw std::runtime_error("Not a valid .spz file");
    }

    uint64_t table_off = is_obs ? hdr.obs_table_offset() : hdr.var_table_offset();
    if (table_off == 0) {
        std::fclose(f);
        return nb::dict{};
    }

    std::fseek(f, static_cast<long>(table_off), SEEK_SET);
    uint8_t tbl_hdr_buf[16];
    if (std::fread(tbl_hdr_buf, 1, 16, f) != 16) {
        std::fclose(f);
        throw std::runtime_error("Failed to read table header");
    }

    ObsVarTableHeader tbl_hdr;
    std::memcpy(tbl_hdr.magic, tbl_hdr_buf, 4);
    std::memcpy(&tbl_hdr.n_rows, tbl_hdr_buf + 4, 4);
    std::memcpy(&tbl_hdr.n_cols, tbl_hdr_buf + 8, 4);
    std::memcpy(&tbl_hdr.header_bytes, tbl_hdr_buf + 12, 4);
    if (!tbl_hdr.valid()) {
        std::fclose(f);
        throw std::runtime_error("Invalid obs/var table magic bytes");
    }

    size_t desc_bytes = static_cast<size_t>(tbl_hdr.n_cols) * sizeof(ColDescriptor);
    std::vector<uint8_t> desc_buf(desc_bytes);
    if (std::fread(desc_buf.data(), 1, desc_bytes, f) != desc_bytes) {
        std::fclose(f);
        throw std::runtime_error("Failed to read table descriptors");
    }

    uint64_t max_end = 0;
    for (uint32_t i = 0; i < tbl_hdr.n_cols; ++i) {
        ColDescriptor desc;
        std::memcpy(&desc, desc_buf.data() + i * sizeof(ColDescriptor), sizeof(ColDescriptor));
        uint64_t col_end = desc.data_offset;
        ColType ct = static_cast<ColType>(desc.col_type);
        switch (ct) {
            case ColType::INT32:   col_end += static_cast<uint64_t>(tbl_hdr.n_rows) * 4; break;
            case ColType::FLOAT32: col_end += static_cast<uint64_t>(tbl_hdr.n_rows) * 4; break;
            case ColType::FLOAT64: col_end += static_cast<uint64_t>(tbl_hdr.n_rows) * 8; break;
            case ColType::BOOL:    col_end += tbl_hdr.n_rows; break;
            case ColType::UINT32:  col_end += static_cast<uint64_t>(tbl_hdr.n_rows) * 4; break;
            case ColType::STRING_DICT:
                col_end += static_cast<uint64_t>(tbl_hdr.n_rows) * 4;
                if (desc.dict_bytes > 0) {
                    uint64_t dict_end = desc.dict_offset + desc.dict_bytes;
                    if (dict_end > max_end) max_end = dict_end;
                }
                break;
        }
        if (col_end > max_end) max_end = col_end;
    }

    size_t hdr_plus_desc = 16 + desc_bytes;
    size_t total_bytes = std::max(hdr_plus_desc, static_cast<size_t>(max_end));
    std::vector<uint8_t> full_buf(total_bytes);
    std::fseek(f, static_cast<long>(table_off), SEEK_SET);
    if (std::fread(full_buf.data(), 1, total_bytes, f) != total_bytes) {
        std::fclose(f);
        throw std::runtime_error("Failed to read table data");
    }
    std::fclose(f);

    auto columns = obs_var_table_deserialize(full_buf.data(), total_bytes);

    nb::dict result;
    for (const auto& col : columns) {
        switch (col.type) {
            case ColType::INT32: {
                nb::list lst;
                for (auto v : col.int_data) {
                    if (v == NA_INT32) lst.append(nb::none());
                    else lst.append(v);
                }
                result[col.name.c_str()] = lst;
                break;
            }
            case ColType::FLOAT32: {
                auto* buf = new double[col.flt_data.size()];
                for (size_t i = 0; i < col.flt_data.size(); ++i)
                    buf[i] = static_cast<double>(col.flt_data[i]);
                nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                size_t shape[1] = {col.flt_data.size()};
                result[col.name.c_str()] = nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
                break;
            }
            case ColType::FLOAT64: {
                auto* buf = new double[col.dbl_data.size()];
                std::memcpy(buf, col.dbl_data.data(), col.dbl_data.size() * sizeof(double));
                nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                size_t shape[1] = {col.dbl_data.size()};
                result[col.name.c_str()] = nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
                break;
            }
            case ColType::BOOL: {
                nb::list lst;
                for (auto v : col.bool_data) {
                    if (v == NA_BOOL) lst.append(nb::none());
                    else lst.append(static_cast<bool>(v));
                }
                result[col.name.c_str()] = lst;
                break;
            }
            case ColType::UINT32: {
                nb::list lst;
                for (auto v : col.uint_data) {
                    if (v == NA_UINT32) lst.append(nb::none());
                    else lst.append(v);
                }
                result[col.name.c_str()] = lst;
                break;
            }
            case ColType::STRING_DICT: {
                nb::list lst;
                for (auto code : col.codes) {
                    if (code == NA_UINT32) lst.append(nb::none());
                    else if (code < col.dict.size()) lst.append(col.dict[code]);
                    else lst.append(nb::none());
                }
                result[col.name.c_str()] = lst;
                break;
            }
        }
    }
    return result;
}

static nb::dict py_st_read_obs(const std::string& path) {
    return py_st_read_table(path, true);
}

static nb::dict py_st_read_var(const std::string& path) {
    return py_st_read_table(path, false);
}

// =============================================================================
// Module definition
// =============================================================================

NB_MODULE(_core, m) {
    m.doc() = "StreamPress C++ extension for .spz sparse matrix I/O";

    // ── Sparse read/write ────────────────────────────────────────────
    m.def("_st_read", &py_st_read,
        "path"_a, "threads"_a = 0, "reorder"_a = true,
        "Read a .spz file, returning CSC components as numpy arrays.");

    m.def("_st_write", &py_st_write,
        "nrow"_a, "ncol"_a, "indptr"_a, "indices"_a, "data"_a,
        "path"_a, "chunk_cols"_a = 0, "verbose"_a = false,
        "include_transpose"_a = true, "use_delta"_a = true,
        "precision"_a = "auto", "threads"_a = 0,
        "obs_dict"_a = nb::none(), "var_dict"_a = nb::none(),
        "value_pred"_a = false, "row_sort"_a = false,
        "chunk_bytes"_a = 0, "transp_chunk_cols"_a = 0,
        "Write a scipy.sparse.csc_matrix to .spz format.");

    m.def("_st_info", &py_st_info,
        "path"_a,
        "Read .spz file metadata without decompressing.");

    m.def("_st_read_transpose", &py_st_read_transpose,
        "path"_a, "threads"_a = 0,
        "Read the pre-stored transpose section of a .spz file.");

    m.def("_st_slice_cols", &py_st_slice_cols,
        "path"_a, "cols"_a, "threads"_a = 0,
        "Read a column range [start, end) from a .spz file.");

    m.def("_st_add_transpose", &py_st_add_transpose,
        "path"_a, "verbose"_a = true,
        "Add a pre-computed transpose section to an existing .spz file.");

    // ── Dense read/write ────────────────────────────────────────
    m.def("_st_write_dense", &py_st_write_dense,
        "X"_a, "path"_a,
        "include_transpose"_a = false, "chunk_cols"_a = 256,
        "codec"_a = 0, "delta"_a = false,
        "Write a dense numpy array to .spz format.");

    m.def("_st_read_dense", &py_st_read_dense,
        "path"_a,
        "Read a dense .spz file into a 2D numpy array.");

    // ── Metadata tables ──────────────────────────────────────────────
    m.def("_st_read_obs", &py_st_read_obs,
        "path"_a,
        "Read obs (row) metadata table from a .spz file.");

    m.def("_st_read_var", &py_st_read_var,
        "path"_a,
        "Read var (column) metadata table from a .spz file.");
}
