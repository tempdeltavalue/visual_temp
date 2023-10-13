#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm.hpp>
#include "gtx/string_cast.hpp"

#include <vector>
#include "common/shaders/Shader.h"

#include <ctime>    


float mouseX = 0.5;
float mouseY = 0.5;


void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        mouseX = xpos;
        mouseY = ypos;
        std::cout << "Cursor Position at (" << xpos << " : " << ypos << std::endl;
    }
}


int main2(int argc, char* argv[])
{

    time_t currentTime;
    time(&currentTime);

    // Convert it to a float
    float currentTimeFloat = static_cast<float>(currentTime);
    std::cout << "Current time (float): " << currentTimeFloat << " seconds since the epoch" << std::endl;

    GLFWwindow* window;

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);


    float aspect_ratio = 16.0 / 9.0;
    int width = 800;

    // Calculate the image height, and ensure that it's at least 1.
    int height = static_cast<int>(width / aspect_ratio);


    //height = (height < 1) ? 1 : height;

    window = glfwCreateWindow(width, height, "temp", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }




    // Camera
    glm::vec3 camera_center = glm::vec3(0.5, 0.5, 0);



    //glm::vec3* pixels = new glm::vec3[width * height]();
    //int blockSize = 64;
    //GLuint pixels_SSBO;
    //glGenBuffers(1, &pixels_SSBO);
    //glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pixels_SSBO);
    //glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec3) * width * height, pixels, GL_DYNAMIC_DRAW);




    //std::string shapeCalculatorPath = "../visual_temp/glsl/ray_tracing/compute_shape.glsl";
    //Shader* shapeShader = new Shader(GL_COMPUTE_SHADER, shapeCalculatorPath);
    //shapeShader->use();
    //shapeShader->setInt("width", width);
    //shapeShader->setInt("height", height);



    std::string fragmentShaderPath = "../visual_temp/glsl/ray_tracing/fragment.glsl";
    Shader* fragmentShader = new Shader(GL_FRAGMENT_SHADER, fragmentShaderPath);
    fragmentShader->setInt("height", height);
    fragmentShader->setInt("width", width);
    fragmentShader->setVec3("camera_center", camera_center);




    GLuint VAO;
    GLuint VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);


    // I need this for fragment shader/////
    float quadVertices[] = {
        // Positions
        -1.0f,  1.0f, 0.0f, // Top-left
         1.0f,  1.0f, 0.0f, // Top-right
         1.0f, -1.0f, 0.0f, // Bottom-right
        -1.0f, -1.0f, 0.0f  // Bottom-left
    };


    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    ////////


    while (!glfwWindowShouldClose(window))
    {

        //Draw
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glClearColor(0.f, 0.f, 0.f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        //shapeShader->use();

        //glDispatchCompute(ceil(width * height / blockSize), 1, 1);
        //glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


        fragmentShader->use();
        fragmentShader->setInt("height", height);
        fragmentShader->setInt("width", width);
        fragmentShader->setVec2("mousePos", glm::vec2(mouseX, mouseY));
        fragmentShader->setVec3("camera_center", camera_center);

        fragmentShader->setFloat("current_time", currentTime);

        
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //delete everything !;
}
