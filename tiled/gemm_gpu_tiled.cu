
// This file contains the code for the implementation
// of a tiled CUDA kernel for GEMM

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

#define TILE_WIDTH 16


void gemm_serial_cpu(const float *A, const float *B, float *C, const int N) {

    // serial gemm for verification

    for (int row = 0; row < N; row ++) {
        for (int col = 0; col < N; col ++){
            float sum = 0.0f;
            for (int k = 0; k < N; k ++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}


void verify_results(const float* cpu_C, const float* gpu_C, const int N) {
    const float epsilon = 1e-4;
    
    for (int i = 0; i < N * N; i++) {
        if (abs(cpu_C[i] - gpu_C[i]) > epsilon) {
            cout << "Verification failed at " << i << endl;
            cout << "CPU result: " << cpu_C[i] << endl;
            cout << "GPU result: " << gpu_C[i] << endl;
            cout << "Absolute difference: " << abs(cpu_C[i] - gpu_C[i]) << endl;
            return;
        }
    }
    cout << "Verification passed" << endl;
}

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C, const int N) {

    // tiled kernel for GEMM

    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    // local and global index
    int t_x = threadIdx.x; // thread x index
    int t_y = threadIdx.y;
    int row = blockIdx.x * TILE_WIDTH + t_x;
    int col = blockIdx.y * TILE_WIDTH + t_y;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE_WIDTH; t ++) {
        s_A[t_x][t_y] = A[row * N + t * TILE_WIDTH + t_y];
        s_B[t_x][t_y] = B[(t * TILE_WIDTH + t_x) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k ++) {
            sum += s_A[t_x][k] * s_B[k][t_y];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;
    const int size = N * N;
    const size_t bytes = size * sizeof(float);


    // host memory
    vector<float> h_A(size);
    vector<float> h_B(size);
    vector<float> h_C_cpu(size);
    vector<float> h_C_gpu(size);

    srand(42);

    for (int i = 0; i < size; i ++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }


    // device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // copy data from host to device
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // kernel
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH); // must match
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    gemm_tiled_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    cout << "Time: " << milliseconds << " ms" << endl;

    // copy data from device to host
    cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // verify
    gemm_serial_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    verify_results(h_C_cpu.data(), h_C_gpu.data(), N);

    // cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}


