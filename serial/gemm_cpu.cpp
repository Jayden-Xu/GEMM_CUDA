
// This file contains the code for a serial implementation of the
// GEMM, served as the baseline

#include <vector>
#include <iostream>
#include <chrono>

using namespace std;


void gemm_serial_cpu(const float *A, const float *B, float *C, const int N) {
    
    // serial gemm via vector representation

    for (int row = 0; row < N; row ++) {
        for (int col = 0; col < N; col ++){
            float sum = 0.0;
            for (int k = 0; k < N; k ++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}


int main() {
    const int N = 1024;
    const int size = N * N;
    vector<float> A(size);
    vector<float> B(size);
    vector<float> C(size);

    srand(42);

    for (int i = 0; i < size; i ++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    cout << "--- CPU Implementation ---" << endl;
    cout << "Matrix Size: " << N << " x " << N << endl;

    // run and time
    auto start = chrono::high_resolution_clock::now();
    gemm_serial_cpu(A.data(), B.data(), C.data(), N);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "CPU version took " << duration.count() << " ms" << endl;
    // use C so that optimizer doesn't get rid of it
    cout << "First element of C: " << C[0] << endl;

    return 0;
}

