#pragma once
#include <vector>
#include <glm.hpp>

// https://stackoverflow.com/questions/38172696/should-i-ever-use-a-vec3-inside-of-a-uniform-buffer-or-shader-storage-buffer-o/38172697#38172697
struct Sphere {
    glm::vec4 center;
    glm::vec4 color;
    glm::vec4 albedo;
    glm::vec4 radius; // float
    glm::vec4 is_reflect; // bool
    glm::vec4 fuzz; // float
    glm::vec4 is_dielectric; // bool

    glm::vec4 is_moving; // bool
    glm::vec4 center2;

};


// https://stackoverflow.com/questions/38172696/should-i-ever-use-a-vec3-inside-of-a-uniform-buffer-or-shader-storage-buffer-o/38172697#38172697
struct Triangle {
    glm::vec4 v1;
    glm::vec4 v2;
    glm::vec4 v3;

    glm::vec4 texCoords;
};


// utils
float getFloatTime() {
    time_t currentTime;
    time(&currentTime);

    return static_cast<float>(currentTime);
}

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
//



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
    sphere3.is_moving = glm::vec4(0, 0, 0, 0);

    Sphere sphere1 = Sphere();
    sphere1.center = glm::vec4(0.42, 0.52, -0.8, 0);
    sphere1.color = glm::vec4(0.5, 0.5, 0.2, 0);
    sphere1.albedo = glm::vec4(0.9, 0.9, 0.9, 0);
    sphere1.radius = glm::vec4(small_radius, 1, 1, 1);
    sphere1.is_reflect = glm::vec4(1, 0, 0, 0);
    sphere1.fuzz = glm::vec4(0.01, 0, 0, 0);
    sphere1.is_dielectric = glm::vec4(0, 0, 0, 0);
    sphere1.is_moving = glm::vec4(0, 0, 0, 0);

    Sphere sphere2 = Sphere();
    sphere2.center = glm::vec4(0.55, 0.58, -0.8, 0);
    sphere2.color = glm::vec4(0.9, 0.2, 0.3, 0);
    sphere2.albedo = glm::vec4(0.2, 0.2, 0.2, 0);
    sphere2.radius = glm::vec4(0.05, 1, 1, 1);
    sphere2.is_reflect = glm::vec4(1, 0, 0, 0);
    sphere2.fuzz = glm::vec4(0.3, 0, 0, 0);
    sphere2.is_dielectric = glm::vec4(0, 0, 0, 0);
    sphere2.is_moving = glm::vec4(0, 0, 0, 0);

    float surface_radius = 100;

    Sphere sphere4 = Sphere();
    sphere4.center = glm::vec4(0.5, 0.5 - surface_radius - small_radius, -1, 0);
    sphere4.color = glm::vec4(0.2, 0.8, 0.2, 0);
    sphere4.albedo = glm::vec4(1, 1, 1, 0);
    sphere4.radius = glm::vec4(surface_radius, 1, 1, 1);
    sphere4.is_reflect = glm::vec4(0, 0, 0, 0);
    sphere4.fuzz = glm::vec4(0, 0, 0, 0);
    sphere4.is_dielectric = glm::vec4(0, 0, 0, 0);
    sphere4.is_moving = glm::vec4(0, 0, 0, 0);

    Sphere sphere5 = Sphere();
    sphere5.center = glm::vec4(0.6, 0.5, -0.5, 0);
    sphere5.color = glm::vec4(1, 1, 1, 0);
    sphere3.albedo = glm::vec4(0.2, 0.2, 0.2, 0);
    sphere5.radius = glm::vec4(small_radius, 1, 1, 1);
    sphere5.is_reflect = glm::vec4(0, 0, 0, 0);
    sphere5.fuzz = glm::vec4(0, 0, 0, 0);
    sphere5.is_dielectric = glm::vec4(1, 0, 0, 0);
    sphere5.is_moving = glm::vec4(0, 0, 0, 0);


    std::vector<Sphere> spheres = { sphere5, sphere3, sphere1, sphere2,  sphere4 }; //, 

    return spheres;
}


std::vector<Sphere>generateRandomSpheres(int spheresCount) {
    std::vector<Sphere> spheres;

    Sphere groundSphere = Sphere();
    float groundRadius = 10;
    groundSphere.center = glm::vec4(0.5, 0.5 - groundRadius - 0.1, -1, 0);

    glm::vec2 seed = glm::vec2(getFloatTime(), getFloatTime() / (1 - getFloatTime()));
    groundSphere.color = glm::vec4(randomVec3(seed, 0., 1), 0);
    groundSphere.albedo = glm::vec4(1, 1, 1, 0);// glm::vec4(randomVec3(0.01, 1), 0);
    groundSphere.radius = glm::vec4(groundRadius, 0, 0, 0);

    groundSphere.is_reflect = glm::vec4(0, 0, 0, 0);
    groundSphere.fuzz = glm::vec4(0, 0, 0, 0);
    groundSphere.is_dielectric = glm::vec4(0, 0, 0, 0);


    for (int i = 0; i < spheresCount; i++) {
        glm::vec2 seed = glm::vec2(glm::pow(i + 1, 3), glm::pow(i + 1, 4));

        Sphere sphere = Sphere();
        sphere.center =  glm::vec4(randomVec3(seed, 0.3, 0.8), 0);
        sphere.center.y += 0.2;

        seed += glm::vec2(2, 2);

        sphere.center.z = -rand(seed, 0.6, 0.9);
        seed += glm::vec2(2, 2);

        sphere.color = glm::vec4(randomVec3(seed, 0., 1), 0);
        seed += glm::vec2(2, 2);

        sphere.albedo = glm::vec4(randomVec3(seed, 0.01, 1), 0);
        seed += glm::vec2(2, 2);

        sphere.radius = glm::vec4(randomVec3(seed, 0.01, 0.05), 0);
        seed += glm::vec2(2, 2);

        sphere.is_reflect = glm::vec4(randomVec3(seed, 0., 1), 0);
        seed += glm::vec2(2, 2);

        sphere.fuzz = glm::vec4(randomVec3(seed, 0., 0.1), 0);
        seed += glm::vec2(2, 2);

        sphere.is_dielectric =  glm::vec4(randomVec3(seed, 0., 1), 0);
        seed += glm::vec2(2, 2);

        sphere.is_dielectric = glm::vec4(randomVec3(seed, 0., 1), 0);
        seed += glm::vec2(2, 2);

        sphere.is_moving = glm::vec4(1, 0, 0, 0); //glm::vec4(randomVec3(initialSeed, 0., 1), 0);
        seed += glm::vec2(2, 2);

        sphere.center2 = sphere.center + glm::vec4(randomVec3(seed, -0.02, 0.02), 0);
        seed += glm::vec2(2, 2);

        spheres.push_back(sphere);
    }

    spheres.push_back(groundSphere);

    return spheres;
}


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


