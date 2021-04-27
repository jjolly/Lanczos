#include "lanczos.h"

void printsparsematrix(SparseMatrix A)
{
  int i, j, k;
  double a;

  for (i = 0; i < A.nrows; i++) {
    printf(" row %6d:\n", i);
    for (k = A.deg[i]; k < A.deg[i + 1]; k++) {
      j = A.adj[k];
      printf("%12d", j);
    }
    printf("\n");
    for (k = A.deg[i]; k < A.deg[i + 1]; k++) {
      a = A.value[k];
      printf("%12.4le", a);
    }
    printf("\n");
    fflush(stdout);
  }                             // end for i
}                               // printsparsematrix
