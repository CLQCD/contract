#include <cuda_runtime.h>
#include <contract.h>

int main(int argc, char *argv[])
{
  init(0);

  void *correl, *propag_i, *propag_j, *propag_m;
  cudaMalloc(&correl, 18 * 24 * 24 * 24 * 2 * sizeof(double));
  cudaMalloc(&propag_i, 18 * 24 * 24 * 24 * 16 * 9 * 2 * sizeof(double));
  cudaMalloc(&propag_j, 18 * 24 * 24 * 24 * 16 * 9 * 2 * sizeof(double));
  cudaMalloc(&propag_m, 18 * 24 * 24 * 24 * 16 * 9 * 2 * sizeof(double));
  proton(correl, propag_i, propag_j, propag_m, IK_JL_MN, 18 * 24 * 24 * 24, 5, 5, 0);
  proton(correl, propag_i, propag_j, propag_m, IK_JN_ML, 18 * 24 * 24 * 24, 5, 5, 0);
  proton(correl, propag_i, propag_j, propag_m, IL_JK_MN, 18 * 24 * 24 * 24, 5, 5, 0);
  proton(correl, propag_i, propag_j, propag_m, IL_JN_MK, 18 * 24 * 24 * 24, 5, 5, 0);
  proton(correl, propag_i, propag_j, propag_m, IN_JK_ML, 18 * 24 * 24 * 24, 5, 5, 0);
  proton(correl, propag_i, propag_j, propag_m, IN_JL_MK, 18 * 24 * 24 * 24, 5, 5, 0);
  cudaFree(correl);
  cudaFree(propag_i);
  cudaFree(propag_j);
  cudaFree(propag_m);

  return 0;
}
