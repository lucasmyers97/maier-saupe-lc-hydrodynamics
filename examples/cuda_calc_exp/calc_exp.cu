#include <iostream>
#include <math.h>
#include <sphere_lebedev_rule.hpp>
#include <chrono>

__global__
void lambdaExp(double *Lambda, double *lebedev_coords,
               unsigned int N, double* lam_exp)
{
  int idx = threadIdx.x;
  int stride = blockDim.x;
  for (unsigned int n = idx; n < N; n += stride)
  {
    double exp_sum{0.0};
    exp_sum += Lambda[1]*lebedev_coords[3*n + 0]*lebedev_coords[3*n + 1];
    exp_sum += Lambda[2]*lebedev_coords[3*n + 0]*lebedev_coords[3*n + 2];
    exp_sum += Lambda[4]*lebedev_coords[3*n + 1]*lebedev_coords[3*n + 2];
    exp_sum *= 2;

    exp_sum += Lambda[0]*lebedev_coords[3*n + 0]*lebedev_coords[3*n + 0];
    exp_sum += Lambda[3]*lebedev_coords[3*n + 1]*lebedev_coords[3*n + 1];
    exp_sum -= (Lambda[0] + Lambda[3])
               *lebedev_coords[3*n + 2]*lebedev_coords[3*n + 2];

    lam_exp[n] = exp(exp_sum);
  }
}

void lambdaExpHost(double *Lambda, double *lebedev_coords,
                   unsigned int N, double* lam_exp)
{
  for (unsigned int n = 0; n < N; ++n)
  {
    double exp_sum{0.0};
    exp_sum += Lambda[1]*lebedev_coords[3*n + 0]*lebedev_coords[3*n + 1];
    exp_sum += Lambda[2]*lebedev_coords[3*n + 0]*lebedev_coords[3*n + 2];
    exp_sum += Lambda[4]*lebedev_coords[3*n + 1]*lebedev_coords[3*n + 2];
    exp_sum *= 2;

    exp_sum += Lambda[0]*lebedev_coords[3*n + 0]*lebedev_coords[3*n + 0];
    exp_sum += Lambda[3]*lebedev_coords[3*n + 1]*lebedev_coords[3*n + 1];
    exp_sum -= (Lambda[0] + Lambda[3])
               *lebedev_coords[3*n + 2]*lebedev_coords[3*n + 2];

    lam_exp[n] = exp(exp_sum);
  }
}

int main()
{
  int order{590};
  double *x, *y, *z, *w;
  x = new double[order];
  y = new double[order];
  z = new double[order];
  w = new double[order];

  ld_by_order(order, x, y, z, w);

  double *lebedev_coords = new double[3*order];
  for (unsigned int i = 0; i < order; ++i)
  {
    lebedev_coords[3*i + 0] = x[i];
    lebedev_coords[3*i + 1] = y[i];
    lebedev_coords[3*i + 2] = z[i];
  }
  delete[] x;
  delete[] y;
  delete[] z;
  delete[] w;

  int vec_dim{5};
  double *Lambda = new double[vec_dim];
  for (int i = 0; i < vec_dim; ++i) {Lambda[i] = 0.0;} 

  double *lambda_exp = new double[order];
  
  double *d_lebedev_coords, *d_Lambda, *d_lambda_exp;
  cudaMalloc(&d_lebedev_coords, 3*order*sizeof(double));
  cudaMalloc(&d_Lambda, vec_dim*sizeof(double));
  cudaMalloc(&d_lambda_exp, order*sizeof(double));

  cudaMemcpy(d_lebedev_coords, lebedev_coords,
             3*order*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Lambda, Lambda, vec_dim*sizeof(double), cudaMemcpyHostToDevice);

  int n_blocks{1};
  int n_threads{32};
  auto start = std::chrono::high_resolution_clock::now();
  lambdaExp<<<n_blocks, n_threads>>>(d_Lambda, d_lebedev_coords,
                                     order, d_lambda_exp);
  cudaDeviceSynchronize();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Operation took: " 
            << duration.count() << " microseconds" << std::endl;

  cudaMemcpy(lambda_exp, d_lambda_exp, order*sizeof(double), cudaMemcpyDeviceToHost);

  start = std::chrono::high_resolution_clock::now();
  lambdaExpHost(Lambda, lebedev_coords, order, lambda_exp);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Operation took: " 
            << duration.count() << " microseconds" << std::endl;

  double sum{0.0};
  for (unsigned int i = 0; i < order; ++i)
  {
    sum += lambda_exp[i];
  }
  std::cout << "Sum is: " << sum << std::endl;

  cudaFree(d_lebedev_coords);
  cudaFree(d_Lambda);
  cudaFree(d_lambda_exp);

  delete[] lebedev_coords;
  delete[] Lambda;
  delete[] lambda_exp;

  return 0;
}
