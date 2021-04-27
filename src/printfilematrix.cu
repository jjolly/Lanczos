//printfilematrix
#include "lanczos.h"
 int printfilematrix(double *diag, FullMatrix V, int k)
{
  FILE * fp;
  int i, j;
  fp = fopen("result", "w");
   for (i = 0; i < k; i++)
    fprintf(fp, "%24.16le", diag[i]);
  fprintf(fp, "\n");
  for (i = 0; i < k; i++)
    fprintf(fp, "%24.16le", diag[k + i]);
  fprintf(fp, "\n");
   for (i = 0; i < V.nrows; i++) {
    for (j = 0; j < k; j++)
      fprintf(fp, "%24.16le", V.value[i + j * V.nrows]);
    fprintf(fp, "\n");
  }
   fclose(fp);
  return 0;
}


