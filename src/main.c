//main
#include "lanczos.h"

int main(int argc, char *argv[])
{
  if (argc != 3) {
    fprintf(stderr, "2 Parameters!\n");
    return 0;
  }

  int i, j, k = strtol(argv[2], NULL, 10), l;

  struct SparseMatrix *A =
      (struct SparseMatrix *)malloc(sizeof(struct SparseMatrix));

  struct FullMatrix *Ap = malloc(sizeof(struct FullMatrix));

  double *res_;

  if (readsparsematrix(argv[1], A) == -1)
    return 0;

  createfullmatrix(Ap, A->nrows, A->ncols);

  for(i = 0; i < Ap->nrows; i++) {
    for(j = 0; j < Ap->ncols; j++) {
      Ap->value[i * Ap->ncols + j] = 0.0;
    }
  }

  for(i = 0; i < A->nrows; i++) {
    l = 0;
    if(A->deg[i] > 0) {
      for(j = 0; j < A->deg[i]; j++) {
        Ap->value[i * A->ncols + A->adj[i][j]] = A->value[i][l];
        l++;
      }
    }
  }

  res_ = lanczos(Ap, k);

  printf("Results saved in result file.\nDiagonal Vector:\n");
  printvector(res_, k);

  free(res_);
  deletesparsematrix(A);
  return 0;
}
