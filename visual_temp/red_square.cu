#include <stdio.h>
#include <iostream>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>
#include <cmath>

#include <curand_kernel.h>

#include <vector>
#include <string> 


#include <chrono>
#include <thread>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "common/utils/stb_image_write.h"

const int WINDOW_WIDTH = 700;
const int WINDOW_HEIGHT = 700;
#define DIM 700

// for cuda usage
#define min(a, b) (a > b)? b: a
#define max(a, b) (a > b)? a: b 



const int U_FIELD = 0;
const int V_FIELD = 1;
const int S_FIELD = 2;


#define CUDA_CHECK_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (cudaSuccess != err) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


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



__device__ float bilinearInterpolation(float* field, float x, float y, float dx, float dy) {
    float h = 0.01; //horizontal resolution of sim
    float grid_sclr = 1.0 / h;


    float offset_x = x - dx;
    float x0 = offset_x * grid_sclr;
    float x1 = x0 + 1;
    float tx = (offset_x - x0 * h) * grid_sclr;

    float offset_y = x - dx;
    float y0 = offset_y * grid_sclr;
    float y1 = y0 + 1;
    float ty = (offset_y - x0 * h) * grid_sclr;



    //int idx00 = x0 * DIM + y0;
    //int idx01 = x1 * DIM + y0;
    //int idx10 = x0 * DIM + y1;
    //int idx11 = x1 * DIM + y1;


    //int real_x0 = static_cast<int>(x0 / h);
    //int real_x1 = static_cast<int>(x1 / h);
    //int real_y0 = static_cast<int>(y0 / h);
    //int real_y1 = static_cast<int>(y1 / h);


    int idx00 = x0 * DIM + y0;
    int idx01 = x1 * DIM + y0;
    int idx10 = x1 * DIM + y1;
    int idx11 = x0 * DIM + y1;


    //int idx00 = x0 + y0 * DIM;
    //int idx01 = x1 + y0 * DIM;
    //int idx10 = x0 + y1 * DIM;
    //int idx11 = x1 + y1 * DIM;

    //printf("%d %d %d %d", idx00, idx01, idx10, idx11);

    float sx = 1.0 - tx;
    float sy = 1.0 - ty;

    float term00 = sx * sy * field[idx00];
    float term01 = dx * sy * field[idx01];
    float term10 = sx * dy * field[idx10];
    float term11 = dx * dy * field[idx11];

    float interpolatedValue = term00 + term01 + term10 + term11;

    return interpolatedValue;
}





// ADVECTION
__device__ float d_sampleField(float* field, float x, float y, float dx, float dy) {
    float h = 0.01;

    float grid_sclr = 1.0 / h;
    float half_cell = 0.5 * h;

    //float dx = 0.0;
    //float dy = 0.0;

    //switch (fieldType) {
    //case U_FIELD: dy = half_cell; break;
    //case V_FIELD: dx = half_cell; break;
    //case S_FIELD: dx = half_cell; dy = half_cell; break;
    //}


    int N = DIM - 1;

    float offset_x = x - dx;
    int x0 = max(static_cast<int>(offset_x * grid_sclr), 0);
    x0 = min(x0, N);
    int x1 = min(x0 + 1, N);
    float tx = (offset_x - x0 * h) * grid_sclr;

    float offset_y = y - dy;
    int y0 = max(static_cast<int>(offset_y * grid_sclr), 0);
    y0 = min(y0, N);
    int y1 = min(y0 + 1, N);
    float ty = (offset_y - y0 * h) * grid_sclr;


    float sx = 1.0 - tx;
    float sy = 1.0 - ty;


    int sidx = x0 * DIM + y0;
    int l_idx = x1 * DIM + y0;
    int b_idx = x1 * DIM + y1;
    int u_idx = x0 * DIM + y1;

    float term00 = sx * sy * field[sidx];
    float term01 = tx * sy * field[l_idx];
    float term10 = tx * ty * field[b_idx];
    float term11 = sx * ty * field[u_idx];

    float adv_value = term00 + term01 + term10 + term11;

    return adv_value;

}


__global__ void advect_velocity(float* u, float* v, float* p, float* s, float* res_u, float* res_v) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int xIndex = idx % DIM;
    int yIndex = idx / DIM;

    int bottomIdx = (yIndex > 0) ? (idx - DIM) : -1;
    int upIdx = (yIndex < DIM - 1) ? (idx + DIM) : -1;
    int leftIdx = (xIndex > 0) ? (idx - 1) : -1;
    int rightIdx = (xIndex < DIM - 1) ? (idx + 1) : -1;


    float h = 0.01; //horizontal resolution of sim
    float half_cell = h * 0.5;

    float dt = 1.f / 60.f; // time step of sim

    float y = xIndex * h;
    float x = yIndex * h;

    float average_v = (v[leftIdx] + v[idx] + v[upIdx] + v[bottomIdx]) / 4.f;
    float average_u = (u[leftIdx] + u[idx] + u[upIdx] + u[bottomIdx]) / 4.f;


    //// U component
    if (s[idx] != 0 && xIndex != 1) {
        //float x = xIndex * h;
        //float y = yIndex * h + half_cell;

        //printf("tsssh %d %d __ %f %f \n", xIndex, yIndex, x, y);


        x = x - dt * average_u;
        y = y - dt * average_v;
        //printf("prev x, y %f %f \n", x, y);

        float result = d_sampleField(u, x, y, 0.0f, half_cell);
        if (result != 0) {
            res_u[idx] = result; //*result;
        }
    }

    // V component
    if (s[idx] != 0.0) {
        //float x = xIndex * h + half_cell;
        //float y = yIndex * h;

        //float average_u = (u[leftIdx] + u[idx] + u[upIdx] + u[bottomIdx]) / 4.f;

        x = x - dt * average_u;
        y = y - dt * average_v;
        float result = d_sampleField(v, x, y, half_cell, 0.f);

        if (result != 0) {
            res_v[idx] = result;
        }
    }

    //printf("VVVVV");

}


__global__ void solve_incompressibility(float* u, float* v, float* d_p, float* s) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int xIndex = idx % DIM;
    int yIndex = idx / DIM;


    int bottomIdx = (yIndex > 0) ? (idx - DIM) : -1;
    int upIdx = (yIndex < DIM - 1) ? (idx + DIM) : -1;
    int leftIdx = (xIndex > 0) ? (idx - 1) : -1;
    int rightIdx = (xIndex < DIM - 1) ? (idx + 1) : -1;


    if (s[idx] != 0) {
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

            float speed = 1;
            int num_iters = 5;

            for (int iter = 0; iter < num_iters; iter++) {

                float div = u[idx] - u[rightIdx] +v[idx] - v[upIdx];
                float p = div / (s_sum);


                u[idx] -= l_s * p * speed;
                u[rightIdx] += r_s * p * speed;

                v[idx] -= b_s * p * speed;
                v[upIdx] += u_s * p * speed;
            }
        }
     }
}

__global__ void advect_mass(float* u, float* v, float*s, float* m, float* res_m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int xIndex = idx % DIM;
    int yIndex = idx / DIM;


    float h = 0.01; //horizontal resolution of sim
    float half_cell = h * 0.5;

    float dt = 1.f / 60.f; // time step of sim

    int bottomIdx = (yIndex > 0) ? (idx - DIM) : -1;
    int upIdx = (yIndex < DIM - 1) ? (idx + DIM) : -1;
    int leftIdx = (xIndex > 0) ? (idx - 1) : -1;
    int rightIdx = (xIndex < DIM - 1) ? (idx + 1) : -1;

    if (s[idx] != 0) {
        float _u = u[idx] + u[rightIdx] * 0.5;
        float _v = v[idx] + u[bottomIdx] * 0.5;

        float x = xIndex * half_cell + dt * _u;
        float y = yIndex * half_cell + dt * _v;
        res_m[idx] = d_sampleField(m, x, y, half_cell, half_cell);

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

    float radius = 50;
    
    // Initialize the array in a loop
    for (int x = 0; x < WINDOW_WIDTH; x++) {
        for (int y = 0; y < WINDOW_HEIGHT; y++) {
            int curr_i = y * WINDOW_WIDTH + x;

            //if (i == WINDOW_HEIGHT - 1) {
            //    h_s[curr_i] = 0.f;
            //}

            if (x == 0 || y == 0 || y == WINDOW_HEIGHT - 1) // edges //  x == WINDOW_WIDTH - 1 ||
                h_s[curr_i] = 0.0;	// solid

            float distance = sqrt(pow(x - 80, 2) + pow(y - 200, 2));
            if (distance <= radius) {
                h_s[curr_i] = 0.0f;
            }

            if (x == 1) {
                //h_s[curr_i] = 0.0;	// solid
                h_u[curr_i] = 1.f;
            }
        }
    }

    std::vector<float> h_m(TOTAL_SIZE, 1.0);

    //why ?
    float pipeH = 0.1 * DIM;
    int minI = floor(0.5 * DIM - 0.5 * pipeH);
    int maxI = floor(0.5 * DIM + 0.5 * pipeH);

    // Assuming h_m is a flattened 2D matrix (1D array)
    for (int i = minI; i < maxI && i < DIM; i++)
        h_m[i * DIM + 3] = 0.0;
    //


    float* d_u;
    float* d_res_u;

    float* d_v;
    float* d_res_v;

    float* d_p;
    float* d_s;

    float* d_m;
    float* d_res_m;

    cudaMalloc(&d_u, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_res_u, TOTAL_SIZE * sizeof(float));

    cudaMalloc(&d_v, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_res_v, TOTAL_SIZE * sizeof(float));

    cudaMalloc(&d_p, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_s, TOTAL_SIZE * sizeof(float));

    cudaMalloc(&d_m, TOTAL_SIZE * sizeof(float));
    cudaMalloc(&d_res_m, TOTAL_SIZE * sizeof(float));


    cudaMemcpy(d_u, h_u.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m.data(), TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);


    //int numThreadsPerBlock = 4; 
    //dim3 blockSize(numThreadsPerBlock, numThreadsPerBlock);
    //dim3 gridSize((DIM + blockSize.x - 1) / blockSize.x, (DIM + blockSize.y - 1) / blockSize.y);
    //printf("vv, x %d %d %d", gridSize.x, gridSize.y, gridSize.z);

    //int threadsPerBlock = 256;
    //int blocksPerGrid = (TOTAL_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    int blockSize = 256;
    int gridSize = (TOTAL_SIZE + blockSize - 1) / blockSize;
    // 
    //yourKernel << <gridSize, blockSize >> > (u, DIM);

    bool fff = true;
    int counter = 0;
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Calculate the elapsed time
            // Introduce a delay of 2 seconds (adjust the duration as needed)

        //if (fff) {
        //    std::chrono::seconds duration(2);
        //    std::this_thread::sleep_for(duration);
        //    fff = false;
        //}

        float time = static_cast<float>(glfwGetTime());

        solve_incompressibility << <gridSize, blockSize >> > (d_u, d_v, d_p, d_s);
        CUDA_CHECK_ERROR();

        d_res_u = d_u;
        d_res_v = d_v;
        advect_velocity << <gridSize, blockSize >> > (d_u, d_v, d_p, d_s, d_res_u, d_res_v);
        CUDA_CHECK_ERROR();
        d_u = d_res_u;
        d_v = d_res_v;


        //d_res_m = d_m;
        //advect_mass << <gridSize, blockSize >> > (d_u, d_v, d_s, d_m, d_res_m);
        //d_m = d_res_m;




        //float* viz_h_m = new float[TOTAL_SIZE];
        //cudaMemcpy(viz_h_m, d_m, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);


        float* viz_h_v = new float[TOTAL_SIZE];
        cudaMemcpy(viz_h_v, d_v, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        float* viz_h_u = new float[TOTAL_SIZE];
        cudaMemcpy(viz_h_u, d_u, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);


        float* viz_h_m = new float[TOTAL_SIZE];
        cudaMemcpy(viz_h_m, d_m, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);


        float* viz_pos_u = new float[TOTAL_SIZE];
        float* viz_neg_u = new float[TOTAL_SIZE];

        //cudaMemcpy(viz_teemp, d_s, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < TOTAL_SIZE; i++) {
            viz_pos_u[i] = 0;
            viz_neg_u[i] = 0;

            if (viz_h_u[i] < 0) {
                viz_neg_u[i] = abs(viz_h_u[i]);
            }
            else {
                viz_pos_u[i] = abs(viz_h_u[i]);
            }

        }

        float max_val = *std::max_element(viz_h_u, viz_h_u + TOTAL_SIZE);
        printf("%f \n", max_val);

        //viz_h_v = normalizeArray(viz_h_v, TOTAL_SIZE);
        //viz_h_u = normalizeArray(viz_h_u, TOTAL_SIZE);

        //cudaMemcpy(d_u, norm_h_u, TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        float* result = new float[TOTAL_SIZE * 3];

        for (int i = 0; i < TOTAL_SIZE; ++i) {
            result[i * 3] = viz_neg_u[i];// viz_teemp[i];

            result[i * 3 + 1] = viz_pos_u[i];// viz_teemp[i];

            result[i * 3 + 2] = 0.0f;
        }

        glClear(GL_COLOR_BUFFER_BIT);

        //glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, norm_h_u); // GL_ALPHA // GL_RGB
        glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_FLOAT, result);
        //glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RED, GL_FLOAT, viz_h_u);


        //counter += 1;
        //unsigned char* pixels = new unsigned char[3 * DIM * DIM]; // RGB image

        //glReadPixels(0, 0, DIM, DIM, GL_RGB, GL_UNSIGNED_BYTE, pixels);

        //std::string name = "images/output" + std::to_string(counter) + ".jpg";
        //stbi_write_jpg(name.c_str(), DIM, DIM, 3, pixels, DIM * 3);


        glfwSwapBuffers(window);
        glfwPollEvents();

        // Clean up host memory
        delete[] viz_h_m;
        delete[] viz_h_v;
        delete[] viz_h_u;
        delete[] result;
        delete[] viz_pos_u;
        delete[] viz_neg_u;

        //delete[] pixels;

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