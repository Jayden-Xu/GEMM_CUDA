
// This file contains the code for the implementation
// of a naive CUDA kernel for GEMM

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void gemm_serial_cpu(const float *A, const float *B, float *C, const int N) {

    // serial gemm, serve as verification

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
        if (fabs(cpu_C[i] - gpu_C[i]) > epsilon) {
            cout << "Verification failed at index " << i << "." << endl;
            cout << "CPU result: " << cpu_C[i] 
                 << ", GPU result: " << gpu_C[i] << endl;
            cout << "Absolute difference: " << fabs(cpu_C[i] - gpu_C[i]) << endl;
            return;
        }
    }

    cout << "Verification passed." << endl;
}


__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, const int N) {

    // naive kernel

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k ++) {
            sum += A[row * N + k] * B[k * N + col];
        }
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
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    cout << " --- Naive GPU Implementation ---" << endl;
    cout << "Matrix Size: " << N << " x " << N << endl;

    // timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    gemm_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    cout << "Naive GPU version took " << milliseconds << " ms" << endl;

    // copy data from device to host
    cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // verify results
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