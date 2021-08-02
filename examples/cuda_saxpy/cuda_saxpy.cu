#include <iostream>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  std::cout << "Made it here" << std::endl;

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  std::cout << "Made to the end of malloc" << std::endl;

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  std::cout << "Made to memcpy" << std::endl;

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  std::cout << "Made to kernel" << std::endl;

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Made past second memcpy" << std::endl;

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
