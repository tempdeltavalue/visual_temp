#include <stdio.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include <cmath>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

const float PI = 3.14159265359f;

// CUDA kernel to fill the RGB buffer with time-based color values
__global__ void fillRGB(unsigned char* rgb, float time) {
    const float PI = 3.14159265359f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = idx * 3;

    // Compute color values based on time
    float frequency = 0.5f;
    float red = 0.5f * sinf(2.0f * PI * frequency * time) + 0.5f;
    float green = 0.5f * sinf(2.0f * PI * frequency * (time + 0.33f)) + 0.5f;
    float blue = 0.5f * sinf(2.0f * PI * frequency * (time + 0.67f)) + 0.5f;

    rgb[offset] = static_cast<unsigned char>(red * 255);     // R value
    rgb[offset + 1] = static_cast<unsigned char>(green * 255); // G value
    rgb[offset + 2] = static_cast<unsigned char>(blue * 255);  // B value
}

int main(int argc, char** argv) {
    GLFWwindow* window;

    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Color Changing Window", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Allocate memory for RGB buffer on device
    unsigned char* dev_rgb;
    cudaMalloc((void**)&dev_rgb, WINDOW_WIDTH * WINDOW_HEIGHT * 3 * sizeof(unsigned char));

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Calculate the elapsed time
        float time = static_cast<float>(glfwGetTime());

        // Fill RGB buffer with time-based color values using kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (WINDOW_WIDTH * WINDOW_HEIGHT + threadsPerBlock - 1) / threadsPerBlock;
        fillRGB << <blocksPerGrid, threadsPerBlock >> > (dev_rgb, time);

        // Allocate memory for RGB buffer on host
        unsigned char* host_rgb = new unsigned char[WINDOW_WIDTH * WINDOW_HEIGHT * 3];

        // Copy RGB buffer from device to host
        cudaMemcpy(host_rgb, dev_rgb, WINDOW_WIDTH * WINDOW_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Render RGB buffer to window
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, host_rgb);
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Clean up host memory
        delete[] host_rgb;
    }

    // Clean up GLFW
    glfwTerminate();

    // Clean up device memory
    cudaFree(dev_rgb);

    return 0;
}