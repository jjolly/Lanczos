#include "lanczos.h"

void printvector(double *x, int k)
{
  int i;

  for (i = 0; i < k; i++)
    printf("%12.4le\n", x[i]);
  printf("\n");
  fflush(stdout);
}                               // end printvec
