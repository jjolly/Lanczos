//deletesparsematrix
#include "lanczos.h"
 void deletesparsematrix(SparseMatrix * A)
{
  int i;
  free(A->value);
   free(A->adj);
  free(A->deg);
  free(A);
} 
