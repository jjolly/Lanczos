//deletesparsematrix
#include "lanczos.h"

void deletesparsematrix(SparseMatrix * A)
{
  free(A->value);
  free(A->adj);
  free(A->deg);
  free(A);
}

