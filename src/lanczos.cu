// lanczos
#include "lanczos.h"
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>

#define checkCudaErrors(call)                                \
 do {                                                        \
   cudaError_t err = call;                                   \
   if (err != cudaSuccess) {                                 \
     printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err));                        \
     exit(EXIT_FAILURE);                                     \
   }                                                         \
 } while (0)

#define NUM_THREAD (256)

__global__
void sum_sqr_vec(int j, int n, double *v, double *result) {
  __shared__ double sum[NUM_THREAD];
  int tid = threadIdx.x;
  int i = tid;
  int offset = j * n;
  sum[tid] = 0.0;
  while(i < n) { 
    sum[tid] += v[offset + i] * v[offset + i];
    if(i + NUM_THREAD < n)
      sum[tid] += v[offset + i + NUM_THREAD] * v[offset + i + NUM_THREAD];
    i += NUM_THREAD * 2;
  }
  __syncthreads();
  if(tid < 128) sum[tid] += sum[tid + 128];
  __syncthreads();
  if(tid <  64) sum[tid] += sum[tid +  64];
  __syncthreads();
  if(tid <  32) sum[tid] += sum[tid + 32];
  if(tid <  16) sum[tid] += sum[tid + 16];
  if(tid <   8) sum[tid] += sum[tid +  8];
  if(tid <   4) sum[tid] += sum[tid +  4];
  if(tid <   2) sum[tid] += sum[tid +  2];
  if(tid <   1) sum[tid] += sum[tid +  1];
  if(tid == 0) *result = sum[0];
}

__global__
void sum_vec_vec(int j, int n, double *v, double *result) {
  __shared__ double sum[NUM_THREAD];
  int tid = threadIdx.x;
  int i = tid;
  int offset1 = j * n;
  int offset2 = (j + 1) * n;
  sum[tid] = 0.0;
  while(i < n) { 
    sum[tid] += v[offset1 + i] * v[offset2 + i];
    if(i + NUM_THREAD < n)
      sum[tid] += v[offset1 + i + NUM_THREAD] * v[offset2 + i + NUM_THREAD];
    i += NUM_THREAD * 2;
  }
  __syncthreads();
  if(tid < 128) sum[tid] += sum[tid + 128];
  __syncthreads();
  if(tid <  64) sum[tid] += sum[tid +  64];
  __syncthreads();
  if(tid <  32) sum[tid] += sum[tid + 32];
  if(tid <  16) sum[tid] += sum[tid + 16];
  if(tid <   8) sum[tid] += sum[tid +  8];
  if(tid <   4) sum[tid] += sum[tid +  4];
  if(tid <   2) sum[tid] += sum[tid +  2];
  if(tid <   1) sum[tid] += sum[tid +  1];
  if(tid == 0) *result = sum[0];
}

__global__
void kernel1(int j, int k, double *sum, double *v, int nrows, int ncols, int *A_deg, int *A_adj,
            double *A_value, double *arrt)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int l;
  double local_sum;
  // WRITE to v[0][0..nrows]
  if(j == 0) v[i] *= rsqrt(*sum);
  local_sum = 0.0;
  if (A_deg[i] < A_deg[i+1])
    for (l = A_deg[i]; l < A_deg[i+1]; l++) {
      // READ from v[j][0..nrows]
      local_sum += v[j * nrows + A_adj[l]] * A_value[l];
    }
  // WRITE to v[j+1][0..nrows]
  v[(j + 1) * nrows + i] = local_sum;

  if (j > 0) {
    // WRITE to v[j+1][0..nrows]
    // READ from arrt[j-1][j]
    // READ from v[j-1][0..nrows]
    v[(j + 1) * nrows + i] +=
        (-arrt[(j - 1) * (k + 1) + j]) * v[(j - 1) * nrows + i];
  }
}
__global__
void kernel2(int j, int k, double *sum, double *v, int nrows, int ncols, int *A_deg, int *A_adj,
            double *A_value, double *arrt)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // WRITE to arrt[j][j]
  arrt[j * (k + 1) + j] = *sum;

  // WRITE to v[j+1][0..nrows]
  // READ from arrt[j][j]
  // READ from v[j][0..nrows]
  v[(j + 1) * nrows + i] += (-arrt[j * (k + 1) + j]) * v[j * nrows + i];
}
__global__
void kernel3(int j, int k, double *sum, double *v, int nrows, int ncols, int *A_deg, int *A_adj,
            double *A_value, double *arrt)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
    // WRITE to arrt[j+1][j]
    if(i == 0) arrt[(j + 1) * (k + 1) + j] = sqrt(*sum);
    // WRITE to v[j+1][0..nrows]
    // READ from arrt[j+1][j]
    v[(j + 1) * nrows + i] *= 1 / arrt[(j + 1) * (k + 1) + j];
    // WRITE to arrt[j][j+1]
    // READ from arrt[j+1][j]
    arrt[j * (k + 1) + j + 1] = arrt[(j + 1) * (k + 1) + j];
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
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  int *d_A_deg, *d_A_adj;
  double *d_v, *d_A_value, *d_arrt, *d_sum;

  checkCudaErrors(cudaMalloc((void **)&d_A_deg, (A->nrows + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_A_adj, A->deg[A->nrows] * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_v, A->nrows * (k + 1) * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_A_value, A->deg[A->nrows] * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_arrt, (k + 1) * (k + 1) * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_sum, sizeof(double)));

  checkCudaErrors(cudaMemcpy(d_A_deg, A->deg, (A->nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_adj, A->adj, A->deg[A->nrows] * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_v, V->value, A->nrows * (k + 1) * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_value, A->value, A->deg[A->nrows] * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_arrt, arrt, (k + 1) * (k + 1) * sizeof(double), cudaMemcpyHostToDevice));
  
  dim3 dimGrid((A->nrows + NUM_THREAD - 1) / NUM_THREAD);
  dim3 dimBlock(NUM_THREAD);

  cudaEventRecord(start, 0);
  sum_sqr_vec<<<1, NUM_THREAD>>>(0, A->nrows, d_v, d_sum);
  for(j = 0; j < k; j++) {
    kernel1<<<dimGrid, dimBlock>>>(j, k, d_sum, d_v, A->nrows, A->ncols, d_A_deg, d_A_adj, d_A_value, d_arrt);
    sum_vec_vec<<<1, NUM_THREAD>>>(j, A->nrows, d_v, d_sum);
    kernel2<<<dimGrid, dimBlock>>>(j, k, d_sum, d_v, A->nrows, A->ncols, d_A_deg, d_A_adj, d_A_value, d_arrt);
    sum_sqr_vec<<<1, NUM_THREAD>>>(j, A->nrows, d_v, d_sum);
    kernel3<<<dimGrid, dimBlock>>>(j, k, d_sum, d_v, A->nrows, A->ncols, d_A_deg, d_A_adj, d_A_value, d_arrt);
  }
  cudaEventRecord(stop, 0);

  cudaMemcpy(arrt, d_arrt, (k + 1) * (k + 1) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(V->value, d_v, A->nrows * (k + 1) * sizeof(double), cudaMemcpyDeviceToHost);
  double sum;
  cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_sum);
  cudaFree(d_arrt);
  cudaFree(d_A_value);
  cudaFree(d_v);
  cudaFree(d_A_adj);
  cudaFree(d_A_deg);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("sum: %lf, Elapsed runtime %fms\n", sum, milliseconds);

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
