cmake_minimum_required(VERSION 3.17)
project(streampress_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find Python + nanobind ───────────────────────────────────────────
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# ── OpenMP (optional) ────────────────────────────────────────────────
find_package(OpenMP QUIET)

# ── Header paths ─────────────────────────────────────────────────────
if(STREAMPRESS_INCLUDE_DIR)
    set(_sp_inc "${STREAMPRESS_INCLUDE_DIR}")
else()
    set(_sp_inc "${CMAKE_CURRENT_SOURCE_DIR}/../include")
endif()

# ── Build the native extension module ────────────────────────────────
nanobind_add_module(_core
    STABLE_ABI
    src/streampress/_bindings.cpp
)

target_include_directories(_core PRIVATE "${_sp_inc}")
target_compile_definitions(_core PRIVATE
    EIGEN_NO_DEBUG
)

# Override nanobind's -Os; sparse codec needs -O3 for speed.
if(MSVC)
    target_compile_options(_core PRIVATE /O2)
else()
    target_compile_options(_core PRIVATE -O3)
    if(NOT DEFINED ENV{CIBUILDWHEEL})
        target_compile_options(_core PRIVATE -march=native)
    endif()
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _core LIBRARY DESTINATION streampress)
