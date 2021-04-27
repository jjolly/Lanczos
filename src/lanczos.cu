// lanczos
#include "lanczos.h"
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>

void kernel(int k, double *v, int nrows, int ncols, int *A_deg, int *A_adj,
            double *A_value, double *arrt)
{
  unsigned int i, j, m;
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
    for (i = 0; i < nrows; i++) {
      if (A_deg[i] < A_deg[i+1])
        for (m = A_deg[i]; m < A_deg[i+1]; m++) {
          // READ from v[j][0..nrows]
          sum += v[j * nrows + A_adj[m]] * A_value[m];
        }
      // WRITE to v[j+1][0..nrows]
      v[(j + 1) * nrows + i] = sum;
      sum = 0.0;
    }

    if (j > 0) {
      for (i = 0; i < nrows; i++) {
        // WRITE to v[j+1][0..nrows]
        // READ from arrt[j-1][j]
        // READ from v[j-1][0..nrows]
        v[(j + 1) * nrows + i] +=
            (-arrt[(j - 1) * (k + 1) + j]) * v[(j - 1) * nrows + i];
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
  double *diag = (double *)malloc(2 * k * sizeof(double));
  struct FullMatrix *V = (struct FullMatrix *)malloc(sizeof(struct FullMatrix));

  createfullmatrix(V, A->nrows, k + 1);
  for (i = 0; i < (k + 1) * A->nrows; i++)
    V->value[i] = (i < A->nrows) ? randomzahl(i) : 0.0;

  arrt = (double *)malloc((k + 1) * (k + 1) * sizeof(double));
  for (i = 0; i < (k + 1) * (k + 1); i++)
    arrt[i] = 0.0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int *d_A_deg, *d_A_adj;
  double *d_v, *d_A_value, *d_arrt;

  d_A_deg = (int *)malloc((A->nrows + 1) * sizeof(int));
  d_A_adj = (int *)malloc(A->deg[A->nrows] * sizeof(int));
  d_v = (double *)malloc(A->nrows * (k + 1) * sizeof(double));
  d_A_value = (double *)malloc(A->deg[A->nrows] * sizeof(double));
  d_arrt = (double *)malloc((k + 1) * (k + 1) * sizeof(double));

  memcpy(d_A_deg, A->deg, (A->nrows + 1) * sizeof(int));
  memcpy(d_A_adj, A->adj, A->deg[A->nrows] * sizeof(int));
  memcpy(d_v, V->value, A->nrows * (k + 1) * sizeof(double));
  memcpy(d_A_value, A->value, A->deg[A->nrows] * sizeof(double));
  memcpy(d_arrt, arrt, (k + 1) * (k + 1) * sizeof(double));

  cudaEventRecord(start, 0);
  kernel(k, d_v, A->nrows, A->ncols, d_A_deg, d_A_adj, d_A_value, d_arrt);
  cudaEventRecord(stop, 0);

  memcpy(arrt, d_arrt, (k + 1) * (k + 1) * sizeof(double));
  memcpy(V->value, d_v, A->nrows * (k + 1) * sizeof(double));

  free(d_arrt);
  free(d_A_value);
  free(d_v);
  free(d_A_adj);
  free(d_A_deg);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Elapsed runtime %fms\n", milliseconds);

  for (j = 0; j < k; j++) {
    diag[j] = arrt[j * (k + 1) + j];
    if (j > 0)
      diag[k + j - 1] = arrt[j * (k + 1) + j - 1];
  }
  diag[2 * k - 1] = arrt[k * (k + 1) + k - 1];

  printfilematrix(diag, *V, k);

  free(arrt);

  deletefullmatrix(V);

  return diag;
}
