#include <stdio.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include <cmath>

#include <curand_kernel.h>

#include <vector>

const int WINDOW_WIDTH = 500;
const int WINDOW_HEIGHT = 500;
#define DIM 500

// for cuda usage
#define min(a, b) (a > b)? b: a
#define max(a, b) (a > b)? a: b 

const int U_FIELD = 0;
const int V_FIELD = 1;
const int S_FIELD = 2;

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


__global__ void solve_incompressibility(float* u, float* v, float* d_p, float* s) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    int xIndex = threadId % DIM;
    int yIndex = threadId / DIM;

    int idx = yIndex * DIM + xIndex;

    if (s[idx] != 0) {

        int upIdx = (yIndex > 0) ? ((yIndex - 1) * DIM + xIndex) : -1;  // -1 indicates no valid neighbor
        int bottomIdx = (yIndex < DIM - 1) ? ((yIndex + 1) * DIM + xIndex) : -1;
        int leftIdx = (xIndex > 0) ? (yIndex * DIM + xIndex - 1) : -1;
        int rightIdx = (xIndex < DIM - 1) ? (yIndex * DIM + xIndex + 1) : -1;


        float l_s = s[leftIdx];
        float r_s = s[rightIdx];
        float u_s = s[upIdx];
        float b_s = s[bottomIdx];

        float s_sum = l_s + r_s + u_s + b_s;

        if (s_sum != 0) {
            // temporarily here
            //float density = 1000;
            //float dt = 1.0 / 50.0;
            //float cp = density * h / dt;

            ////

            int num_iters = 20;
            //u[rightIdx] += u[idx];

            for (int iter = 0; iter < num_iters; iter++) {
                float div = u[rightIdx] - u[idx] + v[bottomIdx] - v[idx];

                float p = - div / s_sum;

                d_p[idx] +=  p; // *cp

                u[idx] -= l_s * p;
                u[rightIdx] += r_s * p;

                v[idx] -= u_s * p;
                v[bottomIdx] += b_s * p;
            }
        }
     }
}



// ADVECTION

__device__ void d_sampleField(int x, int y, int h, int fieldType, float* field, float* result) {

    float h1 = 1.0 / h;
    float h2 = 0.5 * h;

    float dx = 0.0;
    float dy = 0.0;


    switch (fieldType) {
    case U_FIELD: dy = h2; break;
    case V_FIELD: dx = h2; break;
    case S_FIELD: dx = h2; dy = h2; break;
    }

    int x0 = min(max(static_cast<int>((x - dx) * h1), 0), DIM - 1);
    float tx = ((x - dx) - x0 * h) * h1;
    int x1 = min(x0 + 1, DIM - 1);

    int y0 = min(max(static_cast<int>((y - dy) * h1), 0), DIM - 1);
    float ty = ((y - dy) - y0 * h) * h1;
    int y1 = min(y0 + 1, DIM - 1);

    float sx = 1.0 - tx;
    float sy = 1.0 - ty;

    int idx = x * DIM + y;
    result[idx] = sx * sy * field[x0 * DIM + y0] + tx * sy * field[x1 * DIM + y0] + tx * ty * field[x1 * DIM + y1] + sx * ty * field[x0 * DIM + y1];
}


__global__ void advect_velocity(float* u, float* v, float* p, float* s, void (*p_sampleField)(int x, int y, int h, int fieldType, float* field, float* result)) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    int xIndex = threadId % DIM;
    int yIndex = threadId / DIM;

    int idx = yIndex * DIM + xIndex;
    int upIdx = (yIndex > 0) ? ((yIndex - 1) * DIM + xIndex) : -1;  // -1 indicates no valid neighbor
    int bottomIdx = (yIndex < DIM - 1) ? ((yIndex + 1) * DIM + xIndex) : -1;
    int leftIdx = (xIndex > 0) ? (yIndex * DIM + xIndex - 1) : -1;
    int rightIdx = (xIndex < DIM - 1) ? (yIndex * DIM + xIndex + 1) : -1;


    int u_h = 20; //horizontal resolution of sim
    int v_h = 20; //vertical resolution of sim
    float dt = 1.f / 60.f; // time step of sim
    // U component

    if (s[idx] != 0 && s[leftIdx] != 0) {
        int x = xIndex * u_h;
        int y = yIndex * v_h;

        float average_v = (v[leftIdx] + v[idx] + v[(xIndex - 1) * DIM + yIndex + 1] + v[bottomIdx]) / 4.f;

        x = x - dt * u[idx];
        y = y - dt * average_v;

        float* result;
        p_sampleField(x, y, u_h, 0, u, result);
        printf("sampled field %f", result);

    }
}




int main(int argc, char** argv) {
    GLFWwindow* window;

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "temp", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);


    //grid params
    float res = 200;

    float domainHeight = 1.0;
    float domainWidth = domainHeight / WINDOW_WIDTH * WINDOW_HEIGHT;
    float h = domainHeight / res;

    float* d_h;
    cudaMalloc(&d_h, sizeof(float));
    cudaMemcpy(d_h, &h, sizeof(float), cudaMemcpyHostToDevice);



    // Fluid params

    int TOTAL_SIZE = WINDOW_HEIGHT * WINDOW_WIDTH;
    std::vector<float> h_u(TOTAL_SIZE, 0.0);
    std::vector<float> h_v(TOTAL_SIZE, 0.0);
    std::vector<float> h_s(TOTAL_SIZE, 1.0);

    float center = 250;
    float radius = 50;
    
    // Initialize the array in a loop
    for (int i = 0; i < WINDOW_HEIGHT; i++) {
        for (int j = 0; j < WINDOW_WIDTH;j++) {
            int curr_i = i * WINDOW_WIDTH + j;

            //if (i == WINDOW_HEIGHT - 1) {
            //    h_s[curr_i] = 0.f;
            //}

            if (i == 0 || j == 0 || i == WINDOW_WIDTH - 1 || j == WINDOW_WIDTH - 1) // edges
                h_s[curr_i] = 0.0;	// solid

            float distance = sqrt(pow(j - 90, 2) + pow(i - center, 2));
            if (distance <= radius) {
                h_s[curr_i] = 0.0f;
            }

            if (j == 1) {
               h_u[curr_i] = 200.f;
            }
        }
    }


    float* d_u;
    float* d_v;
    float* d_p;
    float* d_s;

    cudaMalloc(&d_u, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_v, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_p, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_s, TOTAL_SIZE * sizeof(float));


    cudaMemcpy(d_u, h_u.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);


    // Define a function pointer to the device function
    void (*sampleFieldFuncPtr)(int x, int y, int h, int fieldType, float* field, float* result); // h, fieldType, field, result
    cudaMemcpyFromSymbol(&sampleFieldFuncPtr, d_sampleField, sizeof(void*));

    int threadsPerBlock = 256;
    int blocksPerGrid = (TOTAL_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Calculate the elapsed time
        float time = static_cast<float>(glfwGetTime());

        solve_incompressibility << <blocksPerGrid, threadsPerBlock >> > (d_u, d_v, d_p, d_s);

        //advect_velocity << <blocksPerGrid, threadsPerBlock >> > (d_u, d_v, d_p, d_s, sampleFieldFuncPtr);

        float* viz_h_v = new float[TOTAL_SIZE];
        cudaMemcpy(viz_h_v, d_v, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        float* viz_h_u = new float[TOTAL_SIZE];
        cudaMemcpy(viz_h_u, d_u, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        //float* norm_h_v = normalizeArray(viz_h_v, TOTAL_SIZE);
        //float* norm_h_u = normalizeArray(viz_h_u, TOTAL_SIZE);

        float* result = new float[TOTAL_SIZE * 3];

        for (int i = 0; i < TOTAL_SIZE; ++i) {
            result[i * 3] = viz_h_v[i];

            result[i * 3 + 1] = viz_h_u[i];

            result[i * 3 + 2] = 0.0f;
        }

        glClear(GL_COLOR_BUFFER_BIT);

        //glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, norm_h_u); // GL_ALPHA // GL_RGB
        glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_FLOAT, result);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Clean up host memory
        delete[] viz_h_v;
        delete[] viz_h_u;
    }

    // Clean up GLFW
    glfwTerminate();

    // Clean up device memory
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_s);
    cudaFree(d_p);

    return 0;
}