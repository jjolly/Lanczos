//createsparsematrix
#include "lanczos.h"

void createsparsematrix(SparseMatrix * A, int m, int n, int *deg)
{
  int i, j;
  A->deg = (int *)malloc(m * sizeof(int));
  j = 0;
  for (i = 0; i < m; i++) {
    A->deg[i] = deg[i];
    j += deg[i];
  }

  A->value = (double *)malloc(j * sizeof(double));

  A->adj = (int *)malloc(j * sizeof(int));
  A->nrows = m;
  A->ncols = n;
}
