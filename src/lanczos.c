// lanczos
#include "lanczos.h"
#include<math.h>

void kernel(int k, double *v, SparseMatrix * A, double *arrt)
{
  unsigned int i, j, l, m;
  double sum;
  double *y, *x;
  sum = 0.0;
  for (i = 0; i < A->nrows; i++) {
    sum += v[i] * v[i];
  }
  sum = 1 / sqrt(sum);
  for (i = 0; i < A->nrows; i++) {
    // WRITE to x[i]
    v[i] *= sum;
  }
  for (j = 0; j < k; j++) {
    y = v + A->nrows * (j + 1);
    x = v + A->nrows * j;
    sum = 0.0;
    l = 0;
    for (i = 0; i < A->nrows; i++) {
      if (A->deg[i] > 0)
        for (m = 0; m < A->deg[i]; m++) {
          sum += x[A->adj[i][m]] * A->value[i][l];
          l++;
        }
      // WRITE to y[i]
      y[i] = sum;
      sum = 0.0;
      l = 0;
    }

    if (j > 0) {
      x = v + A->nrows * (j - 1);
      y = v + A->nrows * (j + 1);
      for (i = 0; i < A->nrows; i++) {
        // WRITE to y[i]
        // READ from arrt[j-1][j]
        y[i] += (-arrt[(j - 1) * (k + 1) + j]) * x[i];
      }
    }
    sum = 0.0;
    x = v + A->nrows * j;
    y = v + A->nrows * (j + 1);
    for (i = 0; i < A->nrows; i++) {
      // READ from y[i]
      sum += x[i] * y[i];
    }
    // WRITE to arrt[j][j]
    arrt[j * (k + 1) + j] = sum;

    x = v + A->nrows * j;
    y = v + A->nrows * (j + 1);
    for (i = 0; i < A->nrows; i++) {
      // WRITE to y[i]
      // READ from arrt[j][j]
      y[i] += (-arrt[j * (k + 1) + j]) * x[i];
    }
    sum = 0.0;
    x = v + A->nrows * (j + 1);
    for (i = 0; i < A->nrows; i++) {
      sum += x[i] * x[i];
    }
    // WRITE to arrt[j+1][j]
    arrt[(j + 1) * (k + 1) + j] = sqrt(sum);
    x = v + A->nrows * (j + 1);
    for (i = 0; i < A->nrows; i++) {
      // READ from arrt[j+1][j]
      x[i] *= 1 / arrt[(j + 1) * (k + 1) + j];
    }
    // WRITE to arrt[j][j+1]
    // READ from arrt[j+1][j]
    arrt[j * (k + 1) + j + 1] = arrt[(j + 1) * (k + 1) + j];
  }
}

double *lanczos(SparseMatrix * A, int k)
{
  int j, i;
  double *arrt;
  double *v = (double *)malloc((k + 1) * A->nrows * sizeof(double));
  //struct Vector *diag = (struct Vector *)malloc(2 * sizeof(struct Vector));
  double *diag = (double *)malloc(2 * k * sizeof(double));
  struct FullMatrix *V = (struct FullMatrix *)malloc(sizeof(struct FullMatrix));

  createfullmatrix(V, A->nrows, k + 1);
  for (i = 0; i < (k + 1) * A->nrows; i++)
    v[i] = 0.0;

  arrt = malloc((k + 1) * (k + 1) * sizeof(double));
  for (i = 0; i < (k + 1) * (k + 1); i++)
    arrt[i] = 0.0;

  for (i = 0; i < A->nrows; i++) {
    v[i] = randomzahl(i);
  }

  kernel(k, v, A, arrt);

  //fuelledia(diag, arrt, k);
  for (j = 0; j < k; j++) {
    diag[j] = arrt[j * (k + 1) + j];
    if (j > 0)
      diag[k + j - 1] = arrt[j * (k + 1) + j - 1];
  }
  diag[2 * k - 1] = arrt[k * (k + 1) + k - 1];

  for (i = 0; i < V->ncols; i++) {
    for (j = 0; j < V->nrows; j++) {
      V->value[j + V->nrows * i] = v[i * A->nrows + j];
    }
  }
  printfilematrix(diag, *V, k);

  free(v);

  free(arrt);

  deletefullmatrix(V);

  return diag;
}
