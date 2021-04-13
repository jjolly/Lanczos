// lanczos
#include "lanczos.h"
#include<math.h>

void kernel(int k, Vector * v, SparseMatrix * A, double *arrt)
{
  unsigned int i, j, l, m;
  double sum;
  Vector *y, *x;
  // scalevector(1 / norm(v), v);
  sum = 0.0;
  for (i = 0; i < v->n; i++) {
    sum += v->value[i] * v->value[i];
  }
  sum = 1 / sqrt(sum);
  for (i = 0; i < v->n; i++) {
    // WRITE to x[i]
    v->value[i] *= sum;
  }
  for (j = 0; j < k; j++) {
    // sparsematrixvector(v+j+1, A, v+j); 
    y = v + j + 1;
    x = v + j;
    sum = 0.0;
    l = 0;
    for (i = 0; i < y->n; i++) {
      if (A->deg[i] > 0)
        for (m = 0; m < A->deg[i]; m++) {
          sum += x->value[A->adj[i][m]] * A->value[i][l];
          l++;
        }
      // WRITE to y[i]
      y->value[i] = sum;
      sum = 0.0;
      l = 0;
    }

    if (j > 0) {
      // axpy(-arrt[j-1][j], v+j-1, v+j+1); 
      x = v + j - 1;
      y = v + j + 1;
      for (i = 0; i < x->n; i++) {
        // WRITE to y[i]
        // READ from arrt[j-1][j]
        y->value[i] += (-arrt[(j - 1) * (k + 1) + j]) * x->value[i];
      }
    }
    // arrt[j][j] = dotproduct(v+j, v+j+1);
    sum = 0.0;
    x = v + j;
    y = v + j + 1;
    for (i = 0; i < x->n; i++) {
      // READ from y[i]
      sum += x->value[i] * y->value[i];
    }
    // WRITE to arrt[j][j]
    arrt[j * (k + 1) + j] = sum;

    // axpy(-arrt[j][j], v+j, v+j+1);
    x = v + j;
    y = v + j + 1;
    for (i = 0; i < x->n; i++) {
      // WRITE to y[i]
      // READ from arrt[j][j]
      y->value[i] += (-arrt[j * (k + 1) + j]) * x->value[i];
    }
    // arrt[j+1][j] = norm(v+j+1);
    sum = 0.0;
    x = v + j + 1;
    for (i = 0; i < x->n; i++) {
      sum += x->value[i] * x->value[i];
    }
    // WRITE to arrt[j+1][j]
    arrt[(j + 1) * (k + 1) + j] = sqrt(sum);
    // scalevector(1/arrt[j+1][j], v+j+1); 
    x = v + j + 1;
    for (i = 0; i < x->n; i++) {
      // READ from arrt[j+1][j]
      x->value[i] *= 1 / arrt[(j + 1) * (k + 1) + j];
    }
    // WRITE to arrt[j][j+1]
    // READ from arrt[j+1][j]
    arrt[j * (k + 1) + j + 1] = arrt[(j + 1) * (k + 1) + j];
  }
}

Vector *lanczos(SparseMatrix * A, int k)
{
  int j, i;
  double *arrt;
  struct Vector *v = (struct Vector *)malloc((k + 1) * sizeof(struct Vector));
  struct Vector *diag = (struct Vector *)malloc(2 * sizeof(struct Vector));
  struct FullMatrix *V = (struct FullMatrix *)malloc(sizeof(struct FullMatrix));

  createvector(diag, k);
  createvector(diag + 1, k);
  createfullmatrix(V, A->nrows, k + 1);
  for (j = 0; j < k + 1; j++) {
    createvector(v + j, A->nrows);
    for (i = 0; i < A->nrows; i++)
      v[j].value[i] = 0.0;
  }

  arrt = malloc((k + 1) * (k + 1) * sizeof(double));
  for (i = 0; i < (k + 1) * (k + 1); i++)
    arrt[i] = 0.0;

  fuellevector(v);

  kernel(k, v, A, arrt);

  fuelledia(diag, arrt, k);
  fillfullmatrix(*V, v);
  printfilematrix(diag, *V, k);

  for (i = 0; i < k + 1; i++)
    free((v + i)->value);
  free(v);

  free(arrt);

  deletefullmatrix(V);

  return diag;
}
