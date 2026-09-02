#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define N 10000000
#define BLOCK_SIZE 256

__global__ void sumKernel(const float* d_input, double* d_output, int n) {
    // Per-thread private accumulation in double
    double threadSum = 0.0;

    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        threadSum += (double)d_input[i];
    }

    // Shared memory for tree reduction
    extern __shared__ double sdata[];
    sdata[threadIdx.x] = threadSum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Block sum is in sdata[0]
    if (threadIdx.x == 0) {
        atomicAdd(d_output, sdata[0]);
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    // Allocate host array
    float* h_input = (float*)malloc(bytes);
    if (!h_input) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Fill host array: fill[i] = (i % 1000) * 0.001f
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)((i % 1000) * 0.001f);
    }

    // Compute CPU reference sum in double
    double cpuSum = 0.0;
    for (int i = 0; i < N; i++) {
        cpuSum += (double)h_input[i];
    }

    // Allocate device arrays
    float* d_input;
    double* d_output;

    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(double)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Initialize output to zero
    double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_output, &zero, sizeof(double), cudaMemcpyHostToDevice));

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t sharedMemSize = BLOCK_SIZE * sizeof(double);

    sumKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_input, d_output, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    double gpuSum = 0.0;
    CUDA_CHECK(cudaMemcpy(&gpuSum, d_output, sizeof(double), cudaMemcpyDeviceToHost));

    // Compute relative error
    double relError = fabs(gpuSum - cpuSum) / fabs(cpuSum);

    printf("CPU sum:  %f\n", cpuSum);
    printf("GPU sum:  %f\n", gpuSum);
    printf("Relative error: %e\n", relError);

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // Free host memory
    free(h_input);

    if (relError < 1e-6) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
