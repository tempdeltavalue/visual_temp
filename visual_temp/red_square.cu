#include <stdio.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include <cmath>

#include <curand_kernel.h>

#include <vector>

const int WINDOW_WIDTH = 1000;
const int WINDOW_HEIGHT = 1000;


float* normalizeArray(const float* array, int size) {
    // Find the minimum and maximum values in the array
    float min_val = *std::min_element(array, array + size);
    float max_val = *std::max_element(array, array + size);

    // Allocate a new array for normalized values
    float* normalizedArray = new float[size];

    // Normalize the array to the range [0, 1]
    for (int i = 0; i < size; ++i) {
        normalizedArray[i] = (array[i] - min_val) / (max_val - min_val);
    }

    return normalizedArray;
}

// CUDA kernel to fill the RGB buffer with time-based color values
__global__ void test_rgb(unsigned char* rgb, float time) {
    const float PI = 3.14159265359f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = idx * 3;

    // Compute color values based on time
    float frequency = 2.f;

    //float myrandf = curand_uniform(dStates + idx + int(time));

    float red = 0.5f * sinf(2.0f * PI * frequency * time) + 0.5f;
    float green = 0.5f * sinf(2.0f * PI * frequency * (time + 0.3)) + 0.5f;
    float blue = 0.5f * sinf(2.0f * PI * frequency * (time + 0.67)) + 0.5f;

    rgb[offset] = static_cast<unsigned char>(red * 255);     // R value
    rgb[offset + 1] = static_cast<unsigned char>(green * 255); // G value
    rgb[offset + 2] = static_cast<unsigned char>(blue * 255);  // B value
}


__global__ void solve_incompressibility(float* u, float* v, float* p, float* s, float *d_h) {
    float h = *d_h;


    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    int leftIdx = idx - 1;
    int rightIdx = idx + 1;
    int upIdx = idx - blockDim.x;
    int bottomIdx = idx + blockDim.x;


    // temporarirly here
    float density = 1000;
    float dt = 1.0 / 1.0;
    float cp = density * h / dt;

    //

    int num_iters = 20;


    for (int iter = 0; iter < num_iters; iter++) {
        if (s[idx] == 0.0)
            continue;

        float mid_s = s[idx];
        float l_s = s[leftIdx];
        float r_s = s[rightIdx];
        float u_s = s[upIdx];
        float b_s = s[bottomIdx];

        float s_sum = l_s + r_s + u_s + b_s;

        if (s_sum == 0.0)
            continue;

        float div = u[rightIdx] - u[idx] + v[bottomIdx] - v[idx];
        float mid_p = -div / s_sum;




        p[idx] += cp * mid_p;

        u[idx] -= l_s * mid_p;

        u[rightIdx] += r_s * mid_p;
        v[idx] -= u_s * mid_p;
        v[bottomIdx] += b_s * mid_p;
    }
}


int main(int argc, char** argv) {
    GLFWwindow* window;

    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "temp", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // test
    unsigned char* dev_rgb;
    cudaMalloc(&dev_rgb, WINDOW_WIDTH * WINDOW_HEIGHT * 3 * sizeof(unsigned char));
    // 

    //grid params
    float res = 200;

    float domainHeight = 1.0;
    float domainWidth = domainHeight / WINDOW_WIDTH * WINDOW_HEIGHT;
    float h = domainHeight / res;

    float* d_h;
    cudaMalloc(&d_h, sizeof(float));
    cudaMemcpy(d_h, &h, sizeof(float), cudaMemcpyHostToDevice);



    // Fluid params

    std::vector<float> h_u(WINDOW_HEIGHT * WINDOW_WIDTH, 0.0);
    std::vector<float> h_v(WINDOW_HEIGHT * WINDOW_WIDTH, 0.0);
    std::vector<float> h_s(WINDOW_HEIGHT * WINDOW_WIDTH, 1.0);

    // Initialize the array in a loop
    for (int i = 0; i < WINDOW_HEIGHT; ++i) {
        for (int j = 0; j < WINDOW_WIDTH; ++j) {
            int curr_i = i * WINDOW_WIDTH + j;

            //h_s[curr_i] = 1.f;


            if (i == 1) {
                h_u[curr_i] = 2.f;
            }
        }
    }


    float* d_u;
    float* d_v;
    float* d_p;
    float* d_s;

    cudaMalloc(&d_u, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float));
    cudaMalloc(&d_v, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float));
    cudaMalloc(&d_p, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float));
    cudaMalloc(&d_s, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float));


    cudaMemcpy(d_u, h_u.data(), WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s.data(), WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);



    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Calculate the elapsed time
        float time = static_cast<float>(glfwGetTime());

        // Fill RGB buffer with time-based color values using kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (WINDOW_WIDTH * WINDOW_HEIGHT + threadsPerBlock - 1) / threadsPerBlock;


        // test
        //test_rgb << <blocksPerGrid, threadsPerBlock >> > (dev_rgb, time);
        //unsigned char* host_rgb = new unsigned char[WINDOW_WIDTH * WINDOW_HEIGHT * 3];
        //cudaMemcpy(host_rgb, dev_rgb, WINDOW_WIDTH * WINDOW_HEIGHT * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        ///

        solve_incompressibility << <blocksPerGrid, threadsPerBlock >> > (d_u, d_v, d_p, d_s, d_h);
        float* h_u = new float[WINDOW_WIDTH * WINDOW_HEIGHT];
        cudaMemcpy(h_u, d_u, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);


        float* norm_h_u = normalizeArray(h_u, WINDOW_WIDTH * WINDOW_HEIGHT);

        //for (int i = 0; i < WINDOW_WIDTH * WINDOW_HEIGHT; ++i) {
        //    norm_h_u[i] *= 255.f;
        //}

        // Render RGB buffer to window
        glClear(GL_COLOR_BUFFER_BIT);
        //glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, norm_h_u); // GL_ALPHA // GL_RGB

        glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RED, GL_FLOAT, norm_h_u);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Clean up host memory
        delete[] h_u;
    }

    // Clean up GLFW
    glfwTerminate();

    // Clean up device memory
    cudaFree(dev_rgb);

    return 0;
}