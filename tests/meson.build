tests = {
  'test_streampress' : files('test_streampress.cpp'),
}

foreach name, src : tests
  exe = executable(name, src,
    include_directories : include_directories('../include'),
    cpp_std : 'c++17',
  )
  test(name, exe)
endforeach
