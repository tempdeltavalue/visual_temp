
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm.hpp>
#include "gtx/string_cast.hpp"
#include <gtc/matrix_transform.hpp>
#include "gtc/random.hpp"

#include <vector>
#include "common/Model.h"

#include <ctime>    

#include "common/shaders/Shader.h"
#include "common/shaders/Shader.h"

#include "common/Mesh.h"
#include "common/Model.h"

#include "common/InputCallbacks.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "common/utils/stb_image_write.h"




// https://stackoverflow.com/questions/38172696/should-i-ever-use-a-vec3-inside-of-a-uniform-buffer-or-shader-storage-buffer-o/38172697#38172697
struct Triangle {
    glm::vec4 v1;
    glm::vec4 v2;
    glm::vec4 v3;

    glm::vec4 texCoords;
};

// https://stackoverflow.com/questions/38172696/should-i-ever-use-a-vec3-inside-of-a-uniform-buffer-or-shader-storage-buffer-o/38172697#38172697
struct Sphere {
    glm::vec4 center;
    glm::vec4 color;
    glm::vec4 albedo;
    glm::vec4 radius; // float
    glm::vec4 is_reflect; // bool
    glm::vec4 fuzz; // float
    glm::vec4 is_dielectric; // bool
};

// compRand3

float rand(glm::vec2 seed) {
    return glm::fract(glm::sin(glm::dot(seed, glm::vec2(12.9898, 78.233))));
}

float rand(glm::vec2 seed, float minVal, float maxVal) {
    //float zeroToOne = 0.5 + 0.5 * rand(seed);
    return minVal + rand(seed) * (maxVal - minVal);
}

glm::vec3 randomVec3(glm::vec2 seed, float min, float max) {
    return glm::vec3(rand(seed, min, max), rand(seed + glm::vec2(1, 1), min, max), rand(seed + glm::vec2(2, 2), min, max));
}

std::vector<Sphere> getFavoriteSpheres() {
    float small_radius = 0.08;
    Sphere sphere3 = Sphere();
    sphere3.center = glm::vec4(0.25, 0.5, -0.8, 0);
    sphere3.color = glm::vec4(0.5, 0.2, 0.7, 1);
    sphere3.albedo = glm::vec4(0.2, 0.2, 0.2, 1);
    sphere3.radius = glm::vec4(small_radius, 1, 1, 1);
    sphere3.is_reflect = glm::vec4(0, 0, 0, 0);
    sphere3.fuzz = glm::vec4(0, 0, 0, 0);
    sphere3.is_dielectric = glm::vec4(0, 0, 0, 0);

    Sphere sphere1 = Sphere();
    sphere3.center = glm::vec4(0.42, 0.52, -0.8, 0);
    sphere3.color = glm::vec4(0.5, 0.5, 0.2, 0);
    sphere3.albedo = glm::vec4(0.9, 0.9, 0.9, 0);
    sphere3.radius = glm::vec4(small_radius, 1, 1, 1);
    sphere3.is_reflect = glm::vec4(1, 0, 0, 0);
    sphere3.fuzz = glm::vec4(0.01, 0, 0, 0);
    sphere3.is_dielectric = glm::vec4(0, 0, 0, 0);

    Sphere sphere2 = Sphere();
    sphere3.center = glm::vec4(0.55, 0.58, -0.8, 0);
    sphere3.color = glm::vec4(0.9, 0.2, 0.3, 0);
    sphere3.albedo = glm::vec4(0.2, 0.2, 0.2, 0);
    sphere3.radius = glm::vec4(0.05, 1, 1, 1);
    sphere3.is_reflect = glm::vec4(1, 0, 0, 0);
    sphere3.fuzz = glm::vec4(0.3, 0, 0, 0);
    sphere3.is_dielectric = glm::vec4(0, 0, 0, 0);

    float surface_radius = 100;

    Sphere sphere4 = Sphere();
    sphere3.center = glm::vec4(0.5, 0.5 - surface_radius - small_radius, -1, 0);
    sphere3.color = glm::vec4(0.2, 0.8, 0.2, 0);
    sphere3.albedo = glm::vec4(1, 1, 1, 0);
    sphere3.radius = glm::vec4(surface_radius, 1, 1, 1);
    sphere3.is_reflect = glm::vec4(0, 0, 0, 0);
    sphere3.fuzz = glm::vec4(0, 0, 0, 0);
    sphere3.is_dielectric = glm::vec4(0, 0, 0, 0);

    Sphere sphere5 = Sphere();
    sphere3.center = glm::vec4(0.6, 0.5, -0.5, 0);
    sphere3.color = glm::vec4(1, 1, 1, 0);
    sphere3.albedo = glm::vec4(0.2, 0.2, 0.2, 0);
    sphere3.radius = glm::vec4(small_radius, 1, 1, 1);
    sphere3.is_reflect = glm::vec4(0, 0, 0, 0);
    sphere3.fuzz = glm::vec4(0, 0, 0, 0);
    sphere3.is_dielectric = glm::vec4(1, 0, 0, 0);


    std::vector<Sphere> spheres =  { sphere5, sphere3, sphere1, sphere2,  sphere4 }; //, 

    return spheres;
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
    int width = 1200;

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
    glfwSetKeyCallback(window, keyCallback);


    //glfwSetCursorPosCallback(window, cursorCallback);
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);


    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }




    // Camera


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


    //Model ourModel("../visual_temp/models/space_rocket/space_rocket.obj", 1);

    //Mesh calcMesh = ourModel.meshes[0];
    //vector<Vertex> vertices = calcMesh.vertices;
    //vector<unsigned int> indices = calcMesh.indices;

    //std::vector<Triangle> triangles;


    //float translateX = 10.f; // Adjust as needed
    //float translateY = 5.f; // Adjust as needed
    //float scale = 0.1f; // Adjust as needed

    // 
    //for (size_t i = 0; i < indices.size(); i += 3) {

    //    glm::vec4 v1 = glm::vec4(vertices[indices[i]].Position, 0);
    //    glm::vec4 v2 = glm::vec4(vertices[indices[i + 1]].Position, 0);
    //    glm::vec4 v3 = glm::vec4(vertices[indices[i + 2]].Position, 0);

    //    v1 = (v1 + glm::vec4(translateX, translateY, 0.0f, 0)) * scale;
    //    v2 = (v2 + glm::vec4(translateX, translateY, 0.0f, 0)) * scale;
    //    v3 = (v3 + glm::vec4(translateX, translateY, 0.0f, 0)) * scale;


    //    v1.z -= 2;
    //    v2.z -= 2;
    //    v3.z -= 2;

    //    Triangle triangle = { v1, v2, v3 , glm::vec4(vertices[indices[i]].TexCoords, 0, 0) };
    //    triangles.push_back(triangle);
    //}





    std::string fragmentShaderPath = "../visual_temp/glsl/ray_tracing/fragment.glsl";
    Shader fragmentShader(GL_FRAGMENT_SHADER, fragmentShaderPath);
    fragmentShader.setInt("height", height);
    fragmentShader.setInt("width", width);
    fragmentShader.setVec3("camera_center", camera_center);
    //fragmentShader.setInt("triangles_len", triangles.size());

    //GLuint pixels_SSBO;
    //glGenBuffers(1, &pixels_SSBO);
    //glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, pixels_SSBO);
    //glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Triangle) * triangles.size(), &triangles[0], GL_DYNAMIC_DRAW);

    //std::cout << "HERE " << ourModel.textures_loaded.size() << std::endl;


    std::vector<Sphere> spheres;// = getFavoriteSpheres();
    std::cout << "tempp " << spheres.size() << std::endl;
    int spheresCount = 5;

    Sphere groundSphere = Sphere();
    float groundRadius = 10;
    groundSphere.center = glm::vec4(0.5, 0.5 - groundRadius - 0.1, -1, 0);

    glm::vec2 initialSeed2 = glm::vec2(currentTimeFloat, currentTimeFloat / (1 - currentTimeFloat));

    groundSphere.color = glm::vec4(randomVec3(initialSeed2, 0., 1), 0);
    groundSphere.albedo = glm::vec4(1, 1, 1, 0);// glm::vec4(randomVec3(0.01, 1), 0);
    groundSphere.radius = glm::vec4(groundRadius, 0, 0, 0);

    groundSphere.is_reflect = glm::vec4(0, 0, 0, 0);
    groundSphere.fuzz = glm::vec4(0, 0, 0, 0);
    groundSphere.is_dielectric = glm::vec4(0, 0, 0, 0);


    for (int i = 0; i < spheresCount; i++) {
        time_t currentTime;
        time(&currentTime);

        // Convert it to a float
        float currentTimeFloat = static_cast<float>(currentTime);

        glm::vec2 initialSeed = glm::vec2(glm::pow(i + 1, 3), glm::pow(i + 1, 4));

        Sphere sphere = Sphere();
        sphere.center =  glm::vec4(randomVec3(initialSeed, 0.4, 0.7), 0);
        //sphere.center.y = 0.5;

        initialSeed += glm::vec2(2, 2);

        sphere.center.z = -rand(initialSeed, 0.7, 0.9);
        initialSeed += glm::vec2(2, 2);

        sphere.color = glm::vec4(randomVec3(initialSeed, 0., 1), 0);
        initialSeed += glm::vec2(2, 2);

        sphere.albedo = glm::vec4(randomVec3(initialSeed, 0.01, 1), 0);
        initialSeed += glm::vec2(2, 2);

        sphere.radius = glm::vec4(randomVec3(initialSeed, 0.01, 0.05), 0);
        initialSeed += glm::vec2(2, 2);

        sphere.is_reflect = glm::vec4(randomVec3(initialSeed, 0., 1), 0);
        initialSeed += glm::vec2(2, 2);

        sphere.fuzz = glm::vec4(randomVec3(initialSeed, 0., 0.1), 0);
        initialSeed += glm::vec2(2, 2);

        sphere.is_dielectric =  glm::vec4(randomVec3(initialSeed, 0., 1), 0);
        initialSeed += glm::vec2(2, 2);

        spheres.push_back(sphere);
    }

    spheres.push_back(groundSphere);

    spheresCount += 1;

    GLuint spheresSSBO;
    glGenBuffers(1, &spheresSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, spheresSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Sphere) * spheres.size(), &spheres[0], GL_DYNAMIC_DRAW);

    while (!glfwWindowShouldClose(window))
    {

        time_t currentTime;
        time(&currentTime);

        // Convert it to a float
        currentTimeFloat = static_cast<float>(currentTime);

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
        fragmentShader.setVec2("mousePos", glm::vec2(mouseX/width, mouseY/height));
        fragmentShader.setVec3("camera_center", camera_center);
        fragmentShader.setFloat("current_time", currentTimeFloat);
        fragmentShader.setInt("spheres_count", spheresCount);



        //fragmentShader.setInt("triangles_len", triangles.size());


        //glActiveTexture(GL_TEXTURE0); // activate proper texture unit before binding
        //// retrieve texture number (the N in diffuse_textureN)

        //fragmentShader.setInt("texture_diffuse1", 0);
        //glBindTexture(GL_TEXTURE_2D, ourModel.textures_loaded[0].id);
        //glActiveTexture(GL_TEXTURE0);



        
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        ////save image
        //unsigned char* pixels = new unsigned char[3 * width * height]; // RGB image

        //glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

        //stbi_write_jpg("output.jpg", width, height, 3, pixels, width * 3);
        //break;

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //delete everything !;
}
