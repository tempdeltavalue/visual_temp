
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

// CUDA kernel to square each element in the array
__global__ void squareArray(int* arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = arr[idx] * arr[idx];
    }
}

int c_main() {
    const int arraySize = 1000;
    const int blockSize = 256;
    const int gridSize = (arraySize + blockSize - 1) / blockSize;

    int* h_array = new int[arraySize];
    int* d_array;

    // Initialize the array with random values
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < arraySize; ++i) {
        h_array[i] = std::rand() % 100;
    }

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_array, arraySize * sizeof(int));

    // Copy the array from the host to the device
    cudaMemcpy(d_array, h_array, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    squareArray << <gridSize, blockSize >> > (d_array, arraySize);

    // Copy the result back from the device to the host
    cudaMemcpy(h_array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] h_array;
    cudaFree(d_array);

    // Print the squared values
    for (int i = 0; i < arraySize; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}