#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm.hpp>
#include "gtx/string_cast.hpp"

#include <vector>
#include "common/Model.h"

#include <ctime>    

#include "common/shaders/Shader.h"
#include "common/shaders/Shader.h"

#include "common/Mesh.h"
#include "common/Model.h"


float mouseX = 0.5;
float mouseY = 0.5;


// https://stackoverflow.com/questions/38172696/should-i-ever-use-a-vec3-inside-of-a-uniform-buffer-or-shader-storage-buffer-o/38172697#38172697
struct Triangle {
    glm::vec4 v1;
    glm::vec4 v2;
    glm::vec4 v3;

    glm::vec4 texCoords;
};



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

float length_squared(glm::vec3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}


glm::vec3 unit_vector(glm::vec3 v) {
    float length = sqrt(length_squared(v));
    return glm::vec3(v.x / length, v.y / length, v.z / length);
}

int main(int argc, char* argv[])
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


    // I need this for fragment shader/////
    float quadVertices[] = {
        // Positions
        -1.0f,  1.0f, 0.0f, // Top-left
         1.0f,  1.0f, 0.0f, // Top-right
         1.0f, -1.0f, 0.0f, // Bottom-right
        -1.0f, -1.0f, 0.0f  // Bottom-left
    };


    // Create and bind the VAO
    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Create and bind the VBO for 'quadVertices'
    GLuint VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    // Define the vertex attribute pointers for quadVertices
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);


    Model ourModel("../visual_temp/models/space_rocket/space_rocket.obj", 1);

    Mesh calcMesh = ourModel.meshes[0];
    vector<Vertex> vertices = calcMesh.vertices;
    vector<unsigned int> indices = calcMesh.indices;

    std::vector<Triangle> triangles;


    float translateX = 10.f; // Adjust as needed
    float translateY = 5.f; // Adjust as needed
    float scale = 0.1f; // Adjust as needed

    // 
    for (size_t i = 0; i < indices.size(); i += 3) {

        glm::vec4 v1 = glm::vec4(vertices[indices[i]].Position, 0);
        glm::vec4 v2 = glm::vec4(vertices[indices[i + 1]].Position, 0);
        glm::vec4 v3 = glm::vec4(vertices[indices[i + 2]].Position, 0);

        v1 = (v1 + glm::vec4(translateX, translateY, 0.0f, 0)) * scale;
        v2 = (v2 + glm::vec4(translateX, translateY, 0.0f, 0)) * scale;
        v3 = (v3 + glm::vec4(translateX, translateY, 0.0f, 0)) * scale;


        v1.z -= 2;
        v2.z -= 2;
        v3.z -= 2;

        Triangle triangle = { v1, v2, v3 , glm::vec4(vertices[indices[i]].TexCoords, 0, 0) };
        triangles.push_back(triangle);
    }




    std::string fragmentShaderPath = "../visual_temp/glsl/ray_tracing/fragment.glsl";
    Shader fragmentShader(GL_FRAGMENT_SHADER, fragmentShaderPath);
    fragmentShader.setInt("height", height);
    fragmentShader.setInt("width", width);
    fragmentShader.setVec3("camera_center", camera_center);
    fragmentShader.setInt("triangles_len", triangles.size());

    GLuint pixels_SSBO;
    glGenBuffers(1, &pixels_SSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, pixels_SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Triangle) * triangles.size(), &triangles[0], GL_DYNAMIC_DRAW);


    std::cout << "HERE " << ourModel.textures_loaded.size() << std::endl;

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
        fragmentShader.use();

        fragmentShader.setInt("height", height);
        fragmentShader.setInt("width", width);
        fragmentShader.setVec2("mousePos", glm::vec2(mouseX, mouseY));
        fragmentShader.setVec3("camera_center", camera_center);
        fragmentShader.setFloat("current_time", currentTime);
        fragmentShader.setInt("triangles_len", triangles.size());


        glActiveTexture(GL_TEXTURE0); // activate proper texture unit before binding
        // retrieve texture number (the N in diffuse_textureN)

        fragmentShader.setInt("texture_diffuse1", 0);
        glBindTexture(GL_TEXTURE_2D, ourModel.textures_loaded[0].id);
        glActiveTexture(GL_TEXTURE0);


        
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //delete everything !;
}
