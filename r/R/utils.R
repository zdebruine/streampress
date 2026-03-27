# Internal helper utilities for streampress

.to_dgCMatrix <- function(x) {
  if (inherits(x, "dgCMatrix")) return(x)
  as(as(x, "CsparseMatrix"), "generalMatrix")
}
