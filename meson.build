project('streampress', 'cpp',
  version : '1.0.0',
  license : 'MIT',
  default_options : [
    'cpp_std=c++17',
    'warning_level=2',
    'optimization=2',
  ],
  meson_version : '>=0.60.0',
)

# ── Header-only library ─────────────────────────────────────────────
streampress_inc = include_directories('include')

streampress_dep = declare_dependency(
  include_directories : streampress_inc,
)

# ── Tests ────────────────────────────────────────────────────────────
build_tests = get_option('tests')
if build_tests
  subdir('tests')
endif
