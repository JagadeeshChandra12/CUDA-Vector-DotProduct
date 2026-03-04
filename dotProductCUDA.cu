#include <iostream>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>

#define N 100000000
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 32


// ---------------- CUDA Kernel ----------------
__global__ void dotProductKernel(double *A, double *B, double *C, long n)
{
    __shared__ double cache[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0.0;

    while (tid < n)
    {
        temp += A[tid] * B[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;

    while (i != 0)
    {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];

        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        C[blockIdx.x] = cache[0];
}


// ---------------- OpenMP Function ----------------
double dotProductOMP(double *A, double *B, long n)
{
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for(long i = 0; i < n; i++)
    {
        sum += A[i] * B[i];
    }

    return sum;
}


// ---------------- CUDA Function ----------------
double dotProductCUDA(double *A, double *B, long n)
{
    double *d_A, *d_B, *d_C;
    double *h_C;

    int threads = THREADS_PER_BLOCK;
    int blocks = BLOCKS_PER_GRID;

    h_C = (double*)malloc(blocks * sizeof(double));

    cudaMalloc(&d_A, N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));
    cudaMalloc(&d_C, blocks * sizeof(double));

    cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dotProductKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaMemcpy(h_C, d_C, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for(int i = 0; i < blocks; i++)
        sum += h_C[i];

    std::cout << "Kernel Time: " << kernelTime << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return sum;
}


// ---------------- Main Function ----------------
int main()
{
    double *A = new double[N];
    double *B = new double[N];

    // Generate random vectors
    for(long i = 0; i < N; i++)
    {
        A[i] = rand() / (double)RAND_MAX;
        B[i] = rand() / (double)RAND_MAX;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    double cpuResult = dotProductOMP(A, B, N);

    auto t1 = std::chrono::high_resolution_clock::now();

    double gpuResult = dotProductCUDA(A, B, N);

    auto t2 = std::chrono::high_resolution_clock::now();

    double cpuTime = std::chrono::duration<double>(t1 - t0).count();
    double gpuTime = std::chrono::duration<double>(t2 - t1).count();

    std::cout << "CPU Result: " << cpuResult << std::endl;
    std::cout << "GPU Result: " << gpuResult << std::endl;

    std::cout << "CPU Time: " << cpuTime << " seconds" << std::endl;
    std::cout << "GPU Time: " << gpuTime << " seconds" << std::endl;

    std::cout << "Speedup: " << cpuTime / gpuTime << std::endl;

    delete[] A;
    delete[] B;

    return 0;

}
