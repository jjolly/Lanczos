//fuelledia
#include "lanczos.h"

/*void fuelledia(Vector * diag, double *arrt, int k)
{
  int j;
  for (j = 0; j < k; j++) {
    diag[0].value[j] = arrt[j * (k + 1) + j];
    if (j > 0)
      diag[1].value[j - 1] = arrt[j * (k + 1) + j - 1];
  }
  diag[1].value[k - 1] = arrt[k * (k + 1) + k - 1];
}*/
