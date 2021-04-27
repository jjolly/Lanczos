#include <stdio.h>
#include <stdlib.h>
#include "lanczos.h"

int readsparsematrix(char *filename, SparseMatrix * A)
{

  FILE *fp;
  int i, l, m, n, *deg;
  fp = fopen(filename, "r");

  if (fp == NULL) {
    fprintf(stderr, "Could not open file %s.\n", filename);
    return -1;
  }
  /* end if */
  fscanf(fp, "%d", &m);
  fscanf(fp, "%d", &n);
  deg = (int *)malloc(m * sizeof(int));
  l = 0;
  for (i = 0; i < m; i++) {
    fscanf(fp, "%d", &(deg[i]));
    l += deg[i];
  }
  createsparsematrix(A, m, n, deg);
  free(deg);

  for (i = 0; i < l; i++) {
    fscanf(fp, "%d", &(A->adj[i]));
  }
  for (i = 0; i < l; ++i) {
    fscanf(fp, "%le", &(A->value[i]));
  }

  fclose(fp);
  return 0;
}                               /* end readsparsematrix */
