// lanczos
#include "lanczos.h"
#include<math.h>
#include<cuda_runtime.h>

void kernel(int k, double *v, int nrows, int *A_deg, int **A_adj, double **A_value, double *arrt)
{
  unsigned int i, j, l, m;
  double sum;
  sum = 0.0;
  for (i = 0; i < nrows; i++) {
    // READ from v[0][0..nrows]
    sum += v[i] * v[i];
  }
  sum = 1 / sqrt(sum);
  for (i = 0; i < nrows; i++) {
    // WRITE to v[0][0..nrows]
    v[i] *= sum;
  }
  for (j = 0; j < k; j++) {
    sum = 0.0;
    l = 0;
    for (i = 0; i < nrows; i++) {
      if (A_deg[i] > 0)
        for (m = 0; m < A_deg[i]; m++) {
          // READ from v[j][0..nrows]
          sum += v[j * nrows + A_adj[i][m]] * A_value[i][l];
          l++;
        }
      // WRITE to v[j+1][0..nrows]
      v[(j + 1) * nrows + i] = sum;
      sum = 0.0;
      l = 0;
    }

    if (j > 0) {
      for (i = 0; i < nrows; i++) {
        // WRITE to v[j+1][0..nrows]
        // READ from arrt[j-1][j]
        // READ from v[j-1][0..nrows]
        v[(j + 1) * nrows + i] += (-arrt[(j - 1) * (k + 1) + j]) * v[(j - 1) * nrows + i];
      }
    }
    sum = 0.0;
    for (i = 0; i < nrows; i++) {
      // READ from v[j][0..nrows]
      // READ from v[j+1][0..nrows]
      sum += v[j * nrows + i] * v[(j + 1) * nrows + i];
    }
    // WRITE to arrt[j][j]
    arrt[j * (k + 1) + j] = sum;

    for (i = 0; i < nrows; i++) {
      // WRITE to v[j+1][0..nrows]
      // READ from arrt[j][j]
      // READ from v[j][0..nrows]
      v[(j + 1) * nrows + i] += (-arrt[j * (k + 1) + j]) * v[j * nrows + i];
    }
    sum = 0.0;
    for (i = 0; i < nrows; i++) {
      // READ from v[j+1][0..nrows]
      sum += v[(j + 1) * nrows + i] * v[(j + 1) * nrows + i];
    }
    // WRITE to arrt[j+1][j]
    arrt[(j + 1) * (k + 1) + j] = sqrt(sum);
    for (i = 0; i < nrows; i++) {
      // WRITE to v[j+1][0..nrows]
      // READ from arrt[j+1][j]
      v[(j + 1) * nrows + i] *= 1 / arrt[(j + 1) * (k + 1) + j];
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

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start,0);
  kernel(k, v, A->nrows, A->deg, A->adj, A->value, arrt);
  cudaEventRecord(stop,0);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Elapsed runtime %f\n", milliseconds);

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
