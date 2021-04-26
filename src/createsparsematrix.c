//createsparsematrix
#include "lanczos.h"
 void createsparsematrix(SparseMatrix * A, int m, int n, int *deg)
{
  int i;
  A->deg = (int *)malloc(m * sizeof(int));
  for (i = 0; i < m; i++)
    A->deg[i] = deg[i];
   A->value = (double *)malloc(n * m * sizeof(double));
   A->adj = (int *)malloc(n * m * sizeof(int));
  A->nrows = m;
  A->ncols = n;
} 
