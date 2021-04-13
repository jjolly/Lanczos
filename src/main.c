//main
#include "lanczos.h"
 int main(int argc, char *argv[])
{
  if (argc != 3) {
    fprintf(stderr, "2 Parameters!\n");
    return 0;
  }
   int i, k = strtol(argv[2], NULL, 10);
   struct SparseMatrix *A =
      (struct SparseMatrix *)malloc(sizeof(struct SparseMatrix));
  double *res_;
   if (readsparsematrix(argv[1], A) == -1)
    return 0;
   res_ = lanczos(A, k);
   printf("Results saved in result file.\nDiagonal Vector:\n");
  printvector(res_, k);
   free(res_);
  deletesparsematrix(A);
   return 0;
}

 
