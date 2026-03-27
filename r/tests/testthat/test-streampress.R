test_that("sparse round-trip preserves values", {
  library(Matrix)
  set.seed(42)
  m <- sparseMatrix(
    i = sample.int(100, 200, replace = TRUE),
    j = sample.int(50,  200, replace = TRUE),
    x = runif(200),
    dims = c(100, 50)
  )
  path <- tempfile(fileext = ".spz")
  st_write(m, path)
  m2 <- st_read(path)
  expect_equal(as.matrix(m), as.matrix(m2), tolerance = 1e-6)
})

test_that("column slice returns correct submatrix", {
  library(Matrix)
  set.seed(1)
  m <- sparseMatrix(
    i = sample.int(200, 500, replace = TRUE),
    j = sample.int(100, 500, replace = TRUE),
    x = runif(500),
    dims = c(200, 100)
  )
  path <- tempfile(fileext = ".spz")
  st_write(m, path, include_transpose = FALSE)

  sub <- st_slice_cols(path, cols = 11L:30L)
  ref <- m[, 11:30, drop = FALSE]
  expect_equal(as.matrix(ref), as.matrix(sub), tolerance = 1e-6)
  expect_equal(ncol(sub), 20L)
})

test_that("st_info returns correct dimensions", {
  library(Matrix)
  set.seed(7)
  m <- sparseMatrix(
    i = sample.int(50, 100, replace = TRUE),
    j = sample.int(20, 100, replace = TRUE),
    x = rpois(100, lambda = 3),
    dims = c(50, 20)
  )
  path <- tempfile(fileext = ".spz")
  st_write(m, path)
  info <- st_info(path)
  expect_equal(info$rows, 50L)
  expect_equal(info$cols, 20L)
})

test_that("obs and var metadata round-trips", {
  library(Matrix)
  set.seed(3)
  m <- sparseMatrix(
    i = sample.int(30, 60, replace = TRUE),
    j = sample.int(10, 60, replace = TRUE),
    x = runif(60),
    dims = c(30, 10)
  )
  obs <- data.frame(id = paste0("obs_", 1:30), stringsAsFactors = FALSE)
  var <- data.frame(id = paste0("var_", 1:10), stringsAsFactors = FALSE)
  path <- tempfile(fileext = ".spz")
  st_write(m, path, obs = obs, var = var)

  obs2 <- st_read_obs(path)
  var2 <- st_read_var(path)
  expect_equal(as.character(obs$id), as.character(obs2$id))
  expect_equal(as.character(var$id), as.character(var2$id))
})

test_that("dense round-trip preserves values", {
  set.seed(11)
  d <- matrix(rnorm(200 * 8), nrow = 200, ncol = 8)
  path <- tempfile(fileext = ".spz")
  st_write_dense(d, path)
  d2 <- st_read_dense(path)
  expect_equal(d, d2, tolerance = 1e-6)
})

test_that("integer matrix is handled correctly", {
  library(Matrix)
  set.seed(99)
  # Integer-valued sparse matrix (counts)
  m <- sparseMatrix(
    i = sample.int(40, 80, replace = TRUE),
    j = sample.int(15, 80, replace = TRUE),
    x = as.double(rpois(80, 2)),
    dims = c(40, 15)
  )
  path <- tempfile(fileext = ".spz")
  st_write(m, path)
  m2 <- st_read(path)
  expect_equal(as.matrix(m), as.matrix(m2), tolerance = 1e-9)
})
